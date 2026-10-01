import logging

from pyspark.sql.functions import current_timestamp, col, to_date

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BronzeToSilverPipeline:
    """Read bronze data, drop duplicates, and write to silver layer."""

    def __init__(self, spark, catalog_name, bronze_schema_name, bronze_table_name,
                 silver_schema_name, silver_table_name):
        self.spark = spark
        self.catalog_name = catalog_name
        self.bronze_schema_name = bronze_schema_name
        self.bronze_table_name = bronze_table_name
        self.silver_schema_name = silver_schema_name
        self.silver_table_name = silver_table_name

    def _bronze_full_name(self):
        return f"{self.catalog_name}.{self.bronze_schema_name}.{self.bronze_table_name}"

    def _silver_full_name(self):
        return f"{self.catalog_name}.{self.silver_schema_name}.{self.silver_table_name}"

    def run(self):
        """Read bronze, drop duplicates, and write to silver."""
        logger.info("Starting bronze-to-silver pipeline...")
        try:
            logger.info(f"Reading bronze table: {self._bronze_full_name()}")
            df = self.spark.table(self._bronze_full_name())
            logger.info("Dropping duplicate rows...")
            deduped_df = df.dropDuplicates()
            casted_df = deduped_df.withColumn("transaction_date", to_date(col('transaction_date')))
            row_count = casted_df.count()
            logger.info(f"{row_count} rows after deduplication.")
            silver_name = self._silver_full_name()
            logger.info(f"Writing to silver: {silver_name}")
            casted_df.withColumn("silver_timestamp", current_timestamp()) \
                .write.format("delta").mode("overwrite") \
                    .option("overwriteSchema", "true") \
                .saveAsTable(silver_name)
            logger.info("Pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise