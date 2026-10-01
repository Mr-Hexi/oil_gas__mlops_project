import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureSplitPipeline:
    """Split feature data into train/test and store in Delta tables + Feature Store."""

    def __init__(self, spark: SparkSession, catalog: str, schema: str,
                 feature_source_table: str, eval_split_table: str,
                 labels_table: str, feature_table: str):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.feature_source_table = feature_source_table
        self.eval_split_table = eval_split_table
        self.labels_table = labels_table
        self.feature_table = feature_table
        self.fe = FeatureEngineeringClient()

    def _full_name(self, table: str) -> str:
        return f"{self.catalog}.{self.schema}.{table}"

    def load_feature_table(self) -> DataFrame:
        """Load the feature engineering output table."""
        full_name = self._full_name(self.feature_source_table)
        logger.info(f"Loading feature source table: {full_name}")
        try:
            df = self.spark.table(full_name)
            logger.info(f"Successfully loaded {full_name}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {full_name}: {e}")
            raise

    def split_by_time(self, df: DataFrame, train_ratio: float = 0.8) -> DataFrame:
        """Add a 'split' column based on a time-based train/test split."""
        logger.info(f"Splitting data with train_ratio={train_ratio}")
        try:
            dates = df.select("transaction_date").distinct().orderBy("transaction_date").collect()
            n = len(dates)
            split_idx = int(n * train_ratio)
            cutoff = dates[split_idx]["transaction_date"]
            logger.info(f"Train/test cutoff date: {cutoff}")

            result = df.withColumn(
                "split",
                F.when(F.col("transaction_date") < F.lit(cutoff), "train").otherwise("test"),
            )
            logger.info("Split column added successfully.")
            return result
        except Exception as e:
            logger.error(f"Time split failed: {e}")
            raise

    def write_split_table(self, df: DataFrame) -> None:
        """Write the split mapping (p_key, transaction_date, split) to a Delta table."""
        full_name = self._full_name(self.eval_split_table)
        logger.info(f"Writing split table: {full_name}")
        try:
            split_df = df.select("p_key", "transaction_date", "split")
            split_df.write.format("delta").mode("overwrite").saveAsTable(full_name)
            logger.info(f"Successfully wrote split table: {full_name}")
        except Exception as e:
            logger.error(f"Failed to write split table {full_name}: {e}")
            raise

    def write_labels_table(self, df: DataFrame) -> None:
        """Write the labels (p_key, transaction_date, total_demand, split) to a Delta table."""
        full_name = self._full_name(self.labels_table)
        logger.info(f"Writing labels table: {full_name}")
        try:
            labels_df = df.select("p_key", "transaction_date", "total_demand", "split")
            labels_df.write.format("delta").mode("overwrite").saveAsTable(full_name)
            logger.info(f"Successfully wrote labels table: {full_name}")
        except Exception as e:
            logger.error(f"Failed to write labels table {full_name}: {e}")
            raise

    def create_feature_store_table(self, df: DataFrame) -> None:
        """Create and write the feature table in the Feature Store."""
        full_name = self._full_name(self.feature_table)
        logger.info(f"Creating feature store table: {full_name}")
        try:
            feature_only_df = df.drop("total_demand", "split")

            self.fe.create_table(
                name=full_name,
                primary_keys=["p_key", "transaction_date"],
                df=feature_only_df,
                description="Demand forecasting features for product-city pairs, daily grain",
                schema=feature_only_df.schema,
                tags={
                    "domain": "oil_gas_demand_forecasting",
                    "grain": "product_city_daily",
                    "use_case": "time_series_forecasting",
                    "source": "sc_gold.feature_demand",
                    "quality": "gold",
                },
                timeseries_column="transaction_date",
            )
            self.fe.write_table(name=full_name, df=feature_only_df, mode="merge")
            logger.info(f"Feature store table created and written: {full_name}")
        except Exception as e:
            logger.error(f"Failed to create feature store table {full_name}: {e}")
            raise

    def run(self) -> None:
        """Run the full split-and-store pipeline."""
        logger.info("Starting feature split pipeline...")
        try:
            df = self.load_feature_table()
            df_split = self.split_by_time(df)
            self.write_split_table(df_split)
            self.write_labels_table(df_split)
            self.create_feature_store_table(df_split)
            logger.info("Pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


# --- Usage example (uncomment to run standalone) ---
# catalog_name = "ct_oil_gas"
# gold_schema_name = "sc_gold"
# fe_table_name = "feature_demand"
# gold_table_name = 'daily_demand'
# eval_split_table_name = "demand_eval_split"
# feature_table_name = "daily_demand_features"
# demand_labels_table_name = "daily_demand_labels"
#
# pipeline = FeatureSplitPipeline(
#     spark=spark,
#     catalog=catalog_name,
#     schema=gold_schema_name,
#     feature_source_table=fe_table_name,
#     eval_split_table=eval_split_table_name,
#     labels_table=demand_labels_table_name,
#     feature_table=feature_table_name,
# )
# pipeline.run()