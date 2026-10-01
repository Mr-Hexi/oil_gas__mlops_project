import logging

from pyspark.sql.functions import col, sum as _sum, avg, round as _round, when

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SilverToGoldPipeline:
    """Aggregate silver transaction data into the gold daily demand table."""

    def __init__(self, spark, catalog_name, silver_schema_name, silver_table_name,
                 gold_schema_name, gold_table_name):
        self.spark = spark
        self.catalog_name = catalog_name
        self.silver_schema_name = silver_schema_name
        self.silver_table_name = silver_table_name
        self.gold_schema_name = gold_schema_name
        self.gold_table_name = gold_table_name

    def _silver_full_name(self):
        return f"{self.catalog_name}.{self.silver_schema_name}.{self.silver_table_name}"

    def _gold_full_name(self):
        return f"{self.catalog_name}.{self.gold_schema_name}.{self.gold_table_name}"

    @staticmethod
    def _dbl(expr, name, digits=2):
        """Round an expression to N digits and cast to double."""
        return _round(expr, digits).cast("double").alias(name)

    def load_silver(self):
        """Load the silver table."""
        full_name = self._silver_full_name()
        logger.info(f"Loading silver table: {full_name}")
        try:
            df = self.spark.table(full_name)
            logger.info(f"Successfully loaded {full_name}")
            return df
        except Exception as e:
            logger.error(f"Failed to load silver table {full_name}: {e}")
            raise

    def aggregate(self, silver_df):
        """Aggregate silver data by product, date, and city into gold-level metrics."""
        logger.info("Aggregating silver data to gold...")
        try:
            gold_df = (
                silver_df.groupBy("product_name", "transaction_date", "destination_city")
                .agg(
                    self._dbl(_sum("demand_quantity"), "total_demand_quantity"),
                    self._dbl(_sum("total_cost_usd"), "total_cost_usd"),
                    self._dbl(avg("supplier_reliability_score"), "avg_supplier_reliability_score"),
                    self._dbl(_sum("ordered_quantity"), "total_ordered_quantity"),
                    self._dbl(_sum("available_inventory"), "total_available_inventory"),
                    self._dbl(avg("unit_price_usd"), "avg_unit_price"),
                    self._dbl(_sum("product_cost_usd"), "total_product_cost_usd"),
                    self._dbl(_sum("transportation_cost_usd"), "total_transportation_cost_usd"),
                    self._dbl(_sum("expected_lead_time_days"), "total_expected_lead_time_days"),
                    self._dbl(_sum("actual_lead_time_days"), "total_actual_lead_time_days"),
                    self._dbl(avg("quality_score"), "avg_quality_score"),
                )
                .withColumn(
                    "total_delay_days",
                    col("total_actual_lead_time_days") - col("total_expected_lead_time_days"),
                )
                .withColumn("is_delayed", when(col("total_delay_days") >= 1, 1).otherwise(0))
                .orderBy("transaction_date")
            )
            logger.info("Aggregation complete.")
            return gold_df
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise

    def write_gold(self, gold_df):
        """Write the gold DataFrame to a Delta table."""
        full_name = self._gold_full_name()
        logger.info(f"Writing gold table: {full_name}")
        try:
            gold_df.write.format("delta").mode("overwrite").saveAsTable(full_name)
            logger.info(f"Successfully wrote gold table: {full_name}")
        except Exception as e:
            logger.error(f"Failed to write gold table {full_name}: {e}")
            raise

    def run(self):
        """Run the full silver-to-gold pipeline."""
        logger.info("Starting silver-to-gold pipeline...")
        try:
            silver_df = self.load_silver()
            gold_df = self.aggregate(silver_df)
            self.write_gold(gold_df)
            logger.info("Pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise