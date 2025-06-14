# Databricks notebook source
# MAGIC %pip install "marvelous@git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0"

# COMMAND ----------

# MAGIC %pip install lightgbm loguru

# COMMAND ----------

# MAGIC %restart_python 

# COMMAND ----------

# imports
from pyspark.sql import SparkSession
import mlflow

import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / "src"))

from src.house_price.config import ProjectConfig
from src.house_price.models.basic_model import BasicModel

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

import pandas as pd
from src.house_price import __version__
from mlflow.utils.environment import _mlflow_conda_env
from marvelous.common import is_databricks


# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
print(config.target)


# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
X_train = train_set[config.num_features + config.cat_features]
y_train = train_set[config.target]

# COMMAND ----------

pipeline = Pipeline(
        steps=[("preprocessor", ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"),
                           config.cat_features)],
            remainder="passthrough")
            ),
               ("regressor", LGBMRegressor(**config.parameters))]
        )

pipeline.fit(X_train, y_train)

# COMMAND ----------

experiment_name = f"{config.experiment_name_basic}"
branch_name = "week2"
run_name = "demo-run-model"
run_desc = "demo run for model logging"
git_sha = "1234567890abcd" # get_git_sha()

mlflow.set_experiment(f"{experiment_name}")
with mlflow.start_run(run_name=run_name,
                      tags={"git_sha": git_sha,
                            "branch": branch_name},
                            description= run_desc) as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(config.parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature
    )

# COMMAND ----------

model_name = f"{config.catalog_name}.{config.schema_name}.{config.model_name_basic}"
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model',
    name=model_name,
    tags={"git_sha": git_sha, "branch": branch_name},)

# COMMAND ----------

# only searching by name is supported
v = mlflow.search_model_versions(
    filter_string=f"name='{model_name}'")
print(v[0].__dict__)

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

# let's set latest-model alias instead
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version = model_version.version,
    )

# COMMAND ----------

model_uri = f"models:/{model_name}@latest-model"
sklearn_pipeline = mlflow.sklearn.load_model(model_uri)
predictions = sklearn_pipeline.predict(X_train[0:1])
print(predictions)

# COMMAND ----------

# delete registered model :
# https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/#delete-a-model-or-model-version
# client = MlflowClient()
# client.delete_registered_model(name="mlops_dev.ineszz46.model_demo")