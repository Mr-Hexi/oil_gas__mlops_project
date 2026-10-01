from Bronze.ingestion_to_bronze import BronzeIngestionPipeline
from Silver.bronze_to_silver import BronzeToSilverPipeline
from Gold.silver_to_gold import SilverToGoldPipeline
import os
import yaml

PROJECT_ROOT = "/Workspace/Users/abuhuzaifakurla@gmail.com/MLops_Project/oil_gas__mlops_project"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs")

def load_config(env="dev"):
    """Load YAML config by environment name (e.g. 'dev', 'stg', 'prod')."""
    config_path = os.path.join(CONFIG_PATH, f"var_{env}.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



config = load_config("dev")

catalog_name = config["catalog_name"]
bronze_schema_name = config["bronze_schema_name"]
bronze_table_name = config["bronze_table_name"]
silver_schema_name = config["silver_schema_name"]
silver_table_name = config["silver_table_name"]
gold_schema_name = config["gold_schema_name"]
gold_table_name = config["gold_table_name"]
file_path = config["file_path"]
drop_cols = config["drop_cols"]



#Ingestion to Bronze
ingestion_pipeline = BronzeIngestionPipeline(
    spark=spark,
    file_path=file_path,
    catalog_name=catalog_name,
    schema_name=bronze_schema_name,
    bronze_table_name=bronze_table_name,
    drop_cols=drop_cols,
)
ingestion_pipeline.run()


## Bronze to Silver
bronze_silver_pipeline = BronzeToSilverPipeline(
    spark=spark,
    catalog_name=catalog_name,
    bronze_schema_name=bronze_schema_name,
    bronze_table_name=bronze_table_name,
    silver_schema_name=silver_schema_name,
    silver_table_name=silver_table_name,
)

bronze_silver_pipeline.run()


# Silver to Gold

silver_to_gold = SilverToGoldPipeline(
    spark=spark,
    catalog_name=catalog_name,
    silver_schema_name=silver_schema_name,
    silver_table_name=silver_table_name,
    gold_schema_name=gold_schema_name,
    gold_table_name=gold_table_name,
)
silver_to_gold.run()