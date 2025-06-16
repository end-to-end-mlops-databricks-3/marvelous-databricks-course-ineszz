# Databricks notebook source
# COMMAND ----------
# MAGIC # %pip install -e ..
# MAGIC %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# MAGIC %pip install lightgbm loguru dotenv

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

# MAGIC %pip  install /Volumes/mlops_dev/ineszz46/data/house_price-1.0.2-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Configure tracking uri
import os
import mlflow
from loguru import logger
from pyspark.sql import SparkSession


from house_price.config import ProjectConfig, Tags
from house_price import __version__ as house_price_v
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from house_price.models.feature_lookup_model import FeatureLookUpModel

from dotenv import load_dotenv
from marvelous.common import is_databricks

# Configure tracking uri
# mlflow.set_tracking_uri("databricks")
# mlflow.set_registry_uri("databricks-uc")
# Default profile:
if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# config_dict = ProjectConfig.from_yaml(config_path="../project_config.yml")
# config_dict["mname_basic"] = "default_basic_value"  # Add default or actual value
# config_dict["mname_custom"] = "default_custom_value"  # Add default or actual value

# config = ProjectConfig(**config_dict)


# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()

# COMMAND ----------

# Define house age feature function
fe_model.define_feature_function()

# COMMAND ----------

# Load data
fe_model.load_data()

# COMMAND ----------

# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()

# COMMAND ----------

# Train the model
fe_model.register_model()

# COMMAND ----------

# Lets run prediction on the last production model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature lookup columns and target
X_test = test_set.drop("OverallQual", "GrLivArea", "GarageCars", config.target)


# COMMAND ----------


from pyspark.sql.functions import col

X_test = X_test.withColumn("LotArea", col("LotArea").cast("int")) \
       .withColumn("OverallCond", col("OverallCond").cast("int")) \
       .withColumn("YearBuilt", col("YearBuilt").cast("int")) \
       .withColumn("YearRemodAdd", col("YearRemodAdd").cast("int")) \
       .withColumn("TotalBsmtSF", col("TotalBsmtSF").cast("int"))


# COMMAND ----------

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# Display predictions
logger.info(predictions)