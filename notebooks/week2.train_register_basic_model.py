# Databricks notebook source
# MAGIC %pip install "marvelous@git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0"

# COMMAND ----------

# MAGIC %pip install lightgbm loguru dotenv

# COMMAND ----------

# MAGIC %restart_python 

# COMMAND ----------

import argparse
import os
import sys
from pathlib import Path

import mlflow
from loguru import logger
from pyspark.sql import SparkSession

sys.path.append(str(Path.cwd().parent / "src"))
from src.house_price.config import ProjectConfig, Tags
from src.house_price.models.basic_model import BasicModel

from dotenv import load_dotenv
from marvelous.common import is_databricks



# COMMAND ----------

# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha": "abcd12345", "branch": "week2", "job_run_id": "12345", "stage": "train"})



# COMMAND ----------

# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)


# COMMAND ----------

basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------

# Train + log the model (runs everything including MLflow logging)
basic_model.train()
basic_model.log_model()

# COMMAND ----------

exp_names = basic_model.experiment_name
runs = mlflow.search_runs(
    experiment_names= [exp_names], 
    filter_string="tags.branch='week2'"
)

if not runs.empty:
    run_id = runs.run_id.iloc[0]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")
else:
    print("No runs found for the specified filter criteria.")

# COMMAND ----------

# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------

# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------

# Register model
basic_model.register_model()

# COMMAND ----------

# Predict on the test set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)
X_test = test_set.drop(config.target).toPandas()
predictions_df = basic_model.load_latest_model_and_predict(X_test)