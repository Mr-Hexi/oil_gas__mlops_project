"""Orchestrator: Feature Engineering → Split → Model Training.

Run this file to execute the full pipeline from feature engineering through model training.
"""

import os
import yaml

from feature_engineering import DemandFeaturePipeline
from split_and_store_feature_table import FeatureSplitPipeline
from model_training import DemandModelTrainer



PROJECT_ROOT = "/Workspace/Users/abuhuzaifakurla@gmail.com/MLops_Project/oil_gas__mlops_project"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs")

def load_config(env="dev"):
    """Load YAML config by environment name (e.g. 'dev', 'stg', 'prod')."""
    config_path = os.path.join(CONFIG_PATH, f"var_{env}.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


    

config = load_config("dev")

catalog_name = config["catalog_name"]
gold_schema_name = config["gold_schema_name"]
gold_table_name = config["gold_table_name"]
gold_feature_table_name = config["gold_feature_table_name"]
fe_table_name = config["fe_table_name"]
feature_table_name = config["feature_table_name"]
eval_split_table_name = config["eval_split_table_name"]
demand_labels_table_name = config["demand_labels_table_name"]
xp_name = config["xp_name"]
current_user = config["current_user"]
experiment_path = f"/Users/{current_user}/{xp_name}"
RANDOM_STATE = config["RANDOM_STATE"]
model_name = config["model_name"]
run_name = config["run_name"]
artifact_name = config["artifact_name"]


# --- Feature engineering ---
pipeline = DemandFeaturePipeline(spark, catalog_name, gold_schema_name)
pipeline.run(gold_table_name, gold_feature_table_name)

# --- Split and save to Feature Store ---
pipeline = FeatureSplitPipeline(
    spark=spark,
    catalog=catalog_name,
    schema=gold_schema_name,
    feature_source_table=fe_table_name,
    eval_split_table=eval_split_table_name,
    labels_table=demand_labels_table_name,
    feature_table=feature_table_name,
)
pipeline.run()

# --- Model Training ---
trainer = DemandModelTrainer(
    spark=spark,
    catalog=catalog_name,
    schema=gold_schema_name,
    feature_table=feature_table_name,
    labels_table=demand_labels_table_name,
    eval_split_table=eval_split_table_name,
    experiment_path=experiment_path,
    random_state=RANDOM_STATE,
)
run_id = trainer.run()