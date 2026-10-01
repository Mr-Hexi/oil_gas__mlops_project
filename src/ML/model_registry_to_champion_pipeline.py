import os
import yaml

from model_register import ModelRegistrationPipeline
from model_predictions import ModelPredictionPipeline
from challenger_validation import ChallengerValidationPipeline
import xgboost


PROJECT_ROOT = "/Workspace/Users/abuhuzaifakurla@gmail.com/MLops_Project/oil_gas__mlops_project"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs")

def load_config(env="dev"):
    """Load YAML config by environment name (e.g. 'dev', 'stg', 'prod')."""
    config_path = os.path.join(CONFIG_PATH, f"var_{env}.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



#Configuration 
config = load_config("dev")

catalog_name = config["catalog_name"]
gold_schema_name = config["gold_schema_name"]
model_name = config["model_name"]

RANDOM_STATE = config["RANDOM_STATE"]

current_user = config["current_user"]
xp_name = config["xp_name"]
run_name = config["run_name"]
artifact_name = config["artifact_name"]

eval_split_table_name = config["eval_split_table_name"]
feature_table_name = config["feature_table_name"]
demand_labels_table_name = config["demand_labels_table_name"]
model_suffix = config["model_suffix"]


# Model Register
model_reg_pipeline = ModelRegistrationPipeline(
    model_name=model_name,
    current_user=current_user,
    experiment_name=xp_name,
    run_name=run_name,
    artifact_name=artifact_name,
)
model_reg_pipeline.run()

#Model prediction 
model_pred_pipeline = ModelPredictionPipeline(
    spark=spark,
    catalog=catalog_name,
    schema=gold_schema_name,
    labels_table=demand_labels_table_name,
    model_suffix=model_suffix,
    r2_threshold=0.8,
)
model_pred_pipeline.run()

#Model validation and champion promotion 
model_val_champ_pipeline = ChallengerValidationPipeline(
    spark=spark,
    catalog=catalog_name,
    schema=gold_schema_name,
    labels_table=demand_labels_table_name,
    model_name=model_name,
)
model_val_champ_pipeline.run()