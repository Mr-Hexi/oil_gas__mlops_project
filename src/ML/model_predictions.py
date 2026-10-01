import logging

import mlflow
import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from sklearn.metrics import r2_score
from mlflow.tracking import MlflowClient
import xgboost

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelPredictionPipeline:
    """Score the best model version on test data and evaluate against a threshold."""

    def __init__(self, spark, catalog, schema, labels_table, model_suffix, r2_threshold=0.8):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.labels_table = labels_table
        self.model_suffix = model_suffix
        self.model_name = f"{catalog}.{schema}.{model_suffix}"
        self.r2_threshold = r2_threshold
        self.fe = FeatureEngineeringClient()
        self.client = MlflowClient()

    def _full_name(self, table):
        return f"{self.catalog}.{self.schema}.{table}"

    def setup_mlflow(self):
        """Set the MLflow registry URI to Unity Catalog."""
        logger.info("Setting up MLflow registry...")
        try:
            mlflow.set_registry_uri("databricks-uc")
            logger.info("MLflow registry set to databricks-uc.")
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            raise

    def find_best_version(self):
        """Find the model version with the highest val_r2 metric."""
        logger.info(f"Searching for best version of {self.model_name}")
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")

            def get_metric(v, key="val_r2"):
                run = self.client.get_run(v.run_id)
                return run.data.metrics.get(key, float("-inf"))

            best = max(versions, key=get_metric)
            best_score = get_metric(best)
            logger.info(f"Best version: {best.version} with val_r2={best_score:.4f}")
            return int(best.version), best_score
        except Exception as e:
            logger.error(f"Failed to find best version: {e}")
            raise

    def score_model_batch(self, model_version):
        """Score the test split and return predictions with labels and R2."""
        logger.info(f"Scoring model version {model_version} on test data...")
        try:
            labels_df = self.spark.table(self._full_name(self.labels_table))
            test_labels_df = labels_df.filter("split == 'test'")
            test_eval_df = test_labels_df.select("p_key", "transaction_date")

            predictions_df = self.fe.score_batch(
                model_uri=f"models:/{self.model_name}/{model_version}",
                df=test_eval_df,
                env_manager="local",
            )

            prediction_with_label_df = predictions_df.select(
                "p_key", "transaction_date", "product_name", "destination_city", "prediction"
            ).join(
                test_labels_df.select("p_key", "transaction_date", "total_demand"),
                on=["p_key", "transaction_date"],
                how="inner",
            )

            n_expected = test_eval_df.count()
            n_actual = prediction_with_label_df.count()
            if n_actual != n_expected:
                raise ValueError(
                    f"Row count mismatch after scoring/join: expected {n_expected}, got {n_actual}."
                )

            pdf = prediction_with_label_df.select("total_demand", "prediction").toPandas()
            test_r2 = r2_score(pdf["total_demand"], pdf["prediction"])
            logger.info(f"Scoring complete. Test R2: {test_r2:.4f}")
            return prediction_with_label_df, test_r2
        except Exception as e:
            logger.error(f"Model scoring failed: {e}")
            raise

    def write_predictions(self, predictions_df, model_version):
        """Write predictions to a Delta table."""
        table_name = self._full_name(f"predictions_with_labels_{self.model_suffix}_{model_version}")
        logger.info(f"Writing predictions to {table_name}")
        try:
            predictions_df.write.format("delta").mode("overwrite").saveAsTable(table_name)
            logger.info(f"Predictions written to {table_name}")
        except Exception as e:
            logger.error(f"Failed to write predictions: {e}")
            raise

    def evaluate_and_tag(self, test_r2, model_version):
        """Check R2 threshold and set tag/alias if passed."""
        logger.info(f"Evaluating test_r2={test_r2:.4f} against threshold={self.r2_threshold}")
        try:
            if test_r2 < self.r2_threshold:
                logger.warning(f"Model FAILED test R2 threshold ({test_r2:.4f} < {self.r2_threshold})")
                return

            if self.is_already_challenger(model_version, test_r2):
                logger.info("Skipping re-tag/alias — equivalent Challenger already registered.")
                return

            self.client.set_model_version_tag(
                name=self.model_name,
                version=model_version,
                key="test_r2",
                value=f"{test_r2:.4f}",
            )
            self.client.set_registered_model_alias(
                name=self.model_name,
                alias="Challenger",
                version=model_version,
            )
            logger.info(f"Model PASSED. Tagged as Challenger (version {model_version}).")
        except Exception as e:
            logger.error(f"Failed to evaluate and tag model: {e}")
            raise
    def is_already_challenger(self, model_version, test_r2, metric_tol=1e-4):
        """Check if the current Challenger alias already points to this version,
        or to a version with an equivalent test_r2."""
        try:
            current = self.client.get_model_version_by_alias(self.model_name, "Challenger")
        except Exception:
            logger.info("No existing Challenger alias found.")
            return False

        if int(current.version) == model_version:
            logger.info(f"Version {model_version} is already tagged Challenger.")
            return True

        current_r2 = current.tags.get("test_r2")
        if current_r2 is not None and abs(float(current_r2) - test_r2) <= metric_tol:
            logger.info(
                f"Current Challenger (version {current.version}) already has an equivalent "
                f"test_r2={current_r2} — skipping re-alias for version {model_version}."
            )
            return True

        return False


    def run(self):
        """Run the full prediction pipeline."""
        logger.info("Starting model prediction pipeline...")
        try:
            self.setup_mlflow()
            model_version, _ = self.find_best_version()
            predictions_df, test_r2 = self.score_model_batch(model_version)
            self.write_predictions(predictions_df, model_version)
            self.evaluate_and_tag(test_r2, model_version)
            logger.info("Pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


# --- Usage example (uncomment to run standalone) ---
# catalog_name = "ct_oil_gas"
# gold_schema_name = "sc_gold"
# eval_split_table_name = "demand_eval_split"
# feature_table_name = "daily_demand_features"
# demand_labels_table_name = "daily_demand_labels"
# model_suffix = "daily_oil_gas_demand_xg_model"
#
# RANDOM_STATE = 42
#
# pipeline = ModelPredictionPipeline(
#     spark=spark,
#     catalog=catalog_name,
#     schema=gold_schema_name,
#     labels_table=demand_labels_table_name,
#     model_suffix=model_suffix,
#     r2_threshold=0.8,
# )
# pipeline.run()