import logging

import mlflow
import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient
from sklearn.metrics import r2_score
from mlflow import MlflowClient

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChallengerValidationPipeline:
    """Validate the Challenger model against the Champion and promote if better."""

    def __init__(self, spark, catalog, schema, labels_table, model_name):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.labels_table = labels_table
        self.model_name = model_name
        self.fe = FeatureEngineeringClient()
        self.client = MlflowClient()

    def _full_name(self, table):
        return f"{self.catalog}.{self.schema}.{table}"

    def get_model_by_alias(self, alias):
        """Get a model version by alias."""
        logger.info(f"Getting model version for alias '{alias}'")
        try:
            model_version = self.client.get_model_version_by_alias(self.model_name, alias)
            version = int(model_version.version)
            logger.info(f"Found {alias} version {version}")
            return model_version, version
        except Exception as e:
            logger.error(f"Failed to get model by alias '{alias}': {e}")
            raise

    def get_val_r2(self, model_version):
        """Get the val_r2 metric from the model's training run."""
        try:
            run = mlflow.get_run(model_version.run_id)
            val_r2 = run.data.metrics["val_r2"]
            logger.info(f"val_r2 for version {model_version.version}: {val_r2:.4f}")
            return val_r2
        except Exception as e:
            logger.error(f"Failed to get val_r2: {e}")
            raise

    def score_model_batch(self, model_alias):
        """Score the test split with the model and return predictions + test R2."""
        logger.info(f"Scoring model with alias '{model_alias}' on test data...")
        try:
            labels_df = self.spark.table(self._full_name(self.labels_table))
            test_labels_df = labels_df.filter("split == 'test'")
            test_eval_df = test_labels_df.select("p_key", "transaction_date")

            predictions_df = self.fe.score_batch(
                model_uri=f"models:/{self.model_name}@{model_alias}",
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

    def validate_and_tag(self, challenger_version, challenger_val_r2, challenger_test_r2):
        """Compare challenger vs champion, set validation tags."""
        logger.info("Validating challenger against champion...")
        try:
            try:
                champion_model, _ = self.get_model_by_alias("Champion")
                champion_val_r2 = self.get_val_r2(champion_model)
                _, champion_test_r2 = self.score_model_batch("Champion")

                metric_val_r2_passed = challenger_val_r2 > champion_val_r2
                metric_test_r2_passed = challenger_test_r2 > champion_test_r2
                logger.info(f"Champion val_r2={champion_val_r2:.4f}, Challenger val_r2={challenger_val_r2:.4f}")
                logger.info(f"Champion test_r2={champion_test_r2:.4f}, Challenger test_r2={challenger_test_r2:.4f}")
            except Exception:
                logger.info("No champion found. Accepting challenger as first model.")
                metric_val_r2_passed = True
                metric_test_r2_passed = True

            self.client.set_model_version_tag(
                name=self.model_name,
                version=str(challenger_version),
                key="metric_val_r2_passed",
                value=metric_val_r2_passed,
            )
            self.client.set_model_version_tag(
                name=self.model_name,
                version=str(challenger_version),
                key="metric_test_r2_passed",
                value=metric_test_r2_passed,
            )
            logger.info(f"Tags set: val_r2_passed={metric_val_r2_passed}, test_r2_passed={metric_test_r2_passed}")
            return metric_val_r2_passed, metric_test_r2_passed
        except Exception as e:
            logger.error(f"Validation and tagging failed: {e}")
            raise

    def promote_to_champion(self, challenger_version, val_passed, test_passed):
        """Promote the challenger to Champion if both metrics passed."""
        logger.info(f"Checking promotion criteria for version {challenger_version}")
        try:
            if val_passed and test_passed:
                self.client.set_registered_model_alias(
                    name=self.model_name,
                    alias="Champion",
                    version=challenger_version,
                )
                logger.info(f"Model version {challenger_version} promoted to Champion!")
            else:
                # raise Exception("Model not ready for promotion")
                logger.info(f"Model version {challenger_version} not ready for promotion")
        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            raise

    def run(self):
        """Run the full validation pipeline."""
        logger.info("Starting challenger validation pipeline...")
        try:
            challenger_model, challenger_version = self.get_model_by_alias("Challenger")
            challenger_val_r2 = self.get_val_r2(challenger_model)
            _, challenger_test_r2 = self.score_model_batch("Challenger")

            val_passed, test_passed = self.validate_and_tag(
                challenger_version, challenger_val_r2, challenger_test_r2
            )
            self.promote_to_champion(challenger_version, val_passed, test_passed)
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
# model_name = "ct_oil_gas.sc_gold.daily_oil_gas_demand_xg_model"
#
# RANDOM_STATE = 42
#
# pipeline = ChallengerValidationPipeline(
#     spark=spark,
#     catalog=catalog_name,
#     schema=gold_schema_name,
#     labels_table=demand_labels_table_name,
#     model_name=model_name,
# )
# pipeline.run()