import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BronzeIngestionPipeline:
    """Read a source Excel file and load it into the bronze layer."""

    def __init__(self, spark: SparkSession, file_path: str, catalog_name: str,
                 schema_name: str, bronze_table_name: str, drop_cols: list):
        self.spark = spark
        self.file_path = file_path
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.bronze_table_name = bronze_table_name
        self.drop_cols = drop_cols

    def _full_name(self):
        return f"{self.catalog_name}.{self.schema_name}.{self.bronze_table_name}"

    def load_to_bronze(self) -> None:
        """Read a source Excel file and overwrite it into the bronze layer."""
        full_name = self._full_name()
        logger.info(f"Loading file: {self.file_path}")
        try:
            df = (
                self.spark.read
                .format("excel")
                .option("headerRows", 1)
                .load(self.file_path)
            )

            if self.drop_cols:
                df = df.drop(*self.drop_cols)

            df_with_ts = df.withColumn("bronze_timestamp", current_timestamp())

            df_with_ts.write \
                .format("delta") \
                .mode("overwrite") \
                .saveAsTable(full_name)

            logger.info(f"Successfully wrote data to {full_name}")
        except Exception as e:
            logger.error(f"Failed to load data to {full_name}: {e}")
            raise

    def run(self) -> None:
        """Run the bronze ingestion pipeline."""
        logger.info("Starting bronze ingestion pipeline...")
        try:
            self.load_to_bronze()
            logger.info("Pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise