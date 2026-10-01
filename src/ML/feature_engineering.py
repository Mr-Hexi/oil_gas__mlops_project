import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DemandFeaturePipeline:
    """Simple pipeline to load gold data, engineer features, and write to a Delta table."""

    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema

    def _full_name(self, table: str) -> str:
        return f"{self.catalog}.{self.schema}.{table}"

    def load_gold_table(self, table: str) -> DataFrame:
        """Load a Delta table from the specified catalog.schema.table."""
        full_name = self._full_name(table)
        logger.info(f"Loading gold table: {full_name}")
        try:
            df = self.spark.table(full_name)
            logger.info(f"Successfully loaded {full_name}")
            return df
        except Exception as e:
            logger.error(f"Failed to load gold table {full_name}: {e}")
            raise

    def _check_duplicates(self, df: DataFrame) -> None:
        """Raise ValueError if duplicate (product_name, destination_city, transaction_date) rows exist."""
        dupe_count = (
            df.groupBy("product_name", "destination_city", "transaction_date")
            .count()
            .filter("count > 1")
            .count()
        )
        if dupe_count > 0:
            raise ValueError(
                f"Found {dupe_count} duplicate rows for (product_name, destination_city, transaction_date)."
            )
        logger.info("Duplicate check passed.")

    def engineer_demand_features(self, df: DataFrame) -> DataFrame:
        """Add lag features, moving average, and calendar features to the demand DataFrame."""
        logger.info("Engineering demand features...")
        try:
            self._check_duplicates(df)

            w = Window.partitionBy("product_name", "destination_city").orderBy("transaction_date")

            result = df.withColumns({
                "p_key": F.sha2(F.concat_ws("_", F.col("product_name"), F.col("destination_city")), 256),
                "demand_lag_1": F.lag("total_demand_quantity", 1).over(w),
                "demand_lag_7": F.lag("total_demand_quantity", 7).over(w),
                "demand_ma_7": F.avg("total_demand_quantity").over(w.rowsBetween(-6, 0)),
                "day_of_week": F.dayofweek("transaction_date"),
                "month": F.month("transaction_date"),
            }).select(
                "p_key",
                "transaction_date",
                "product_name",
                "destination_city",
                F.col("total_demand_quantity").alias("total_demand"),
                "avg_unit_price",
                F.col("total_available_inventory").alias("total_inventory"),
                "demand_lag_1",
                "demand_lag_7",
                "demand_ma_7",
                "day_of_week",
                "month",
            )

            logger.info("Feature engineering complete.")
            return result
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

    def write_feature_table(self, df: DataFrame, table: str) -> None:
        """Write the feature DataFrame to a Delta table (overwrite mode)."""
        full_name = self._full_name(table)
        logger.info(f"Writing feature table: {full_name}")
        try:
            df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable(full_name)
            logger.info(f"Successfully wrote feature table: {full_name}")
        except Exception as e:
            logger.error(f"Failed to write feature table {full_name}: {e}")
            raise

    def add_constraints(self, table: str) -> None:
        """Add NOT NULL and PRIMARY KEY constraints to the feature table."""
        full_name = self._full_name(table)
        logger.info(f"Adding constraints to {full_name}")
        try:
            self.spark.sql(f"ALTER TABLE {full_name} ALTER COLUMN p_key SET NOT NULL")
            self.spark.sql(f"ALTER TABLE {full_name} ALTER COLUMN transaction_date SET NOT NULL")
            self.spark.sql(f"ALTER TABLE {full_name} ADD CONSTRAINT {table}_pk PRIMARY KEY (p_key, transaction_date)")
            logger.info(f"Constraints added successfully to {full_name}")
        except Exception as e:
            logger.error(f"Failed to add constraints to {full_name}: {e}")
            raise

    def run(self, gold_table: str, feature_table: str) -> None:
        """Run the full pipeline: load, engineer, write, and add constraints."""
        logger.info("Starting demand feature pipeline...")
        try:
            df = self.load_gold_table(gold_table)
            df_features = self.engineer_demand_features(df)
            self.write_feature_table(df_features, feature_table)
            self.add_constraints(feature_table)
            logger.info("Pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


# --- Usage example (uncomment to run standalone) ---
# catalog_name = "ct_oil_gas"
# gold_schema_name = "sc_gold"
# gold_table_name = "daily_demand"
# feature_table_name = "feature_demand"
#
# pipeline = DemandFeaturePipeline(spark, catalog_name, gold_schema_name)
# pipeline.run(gold_table_name, feature_table_name)