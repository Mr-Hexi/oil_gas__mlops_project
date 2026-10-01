import logging

import mlflow
from mlflow.tracking import MlflowClient
import xgboost

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelRegistrationPipeline:
    """Find the best training run and register the model in Unity Catalog."""

    def __init__(self, model_name, current_user, experiment_name, run_name, artifact_name):
        self.model_name = model_name
        self.current_user = current_user
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.artifact_name = artifact_name
        self.client = MlflowClient()

    def setup_mlflow(self):
        """Set the MLflow registry URI to Unity Catalog."""
        logger.info("Setting up MLflow registry...")
        try:
            mlflow.set_registry_uri("databricks-uc")
            logger.info("MLflow registry set to databricks-uc.")
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            raise

    def find_best_run(self):
        """Search the experiment for the best run by val_r2 metric."""
        xp_path = f"/Users/{self.current_user}"
        logger.info(f"Finding best run from {self.experiment_name} in {xp_path}")
        try:
            experiment_id = mlflow.search_experiments(
                filter_string=f"name LIKE '{xp_path}/{self.experiment_name}%'",
                order_by=["last_update_time DESC"],
            )[0].experiment_id

            best_model = mlflow.search_runs(
                experiment_ids=experiment_id,
                order_by=["metrics.val_r2 DESC"],
                max_results=1,
                filter_string=f"status = 'FINISHED' and run_name = '{self.run_name}'",
            )

            if best_model.empty:
                raise ValueError("No finished runs found matching the criteria.")

            run_id = best_model.iloc[0]["run_id"]
            best_score = best_model["metrics.val_r2"].values[0]
            logger.info(f"Best run found: run_id={run_id}, val_r2={best_score:.4f}")
            return run_id, best_score
        except Exception as e:
            logger.error(f"Failed to find best run: {e}")
            raise

    def is_duplicate_model(self, run_id, best_score, metric_key="val_r2", param_tol=0, metric_tol=1e-4):
        """
        Check if an already-registered version has the same hyperparameters and a
        near-identical metric value as the candidate run. Returns True if a duplicate is found.
        """
        logger.info(f"Checking for duplicate versions of {self.model_name}")
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
        except Exception as e:
            logger.warning(f"Could not check existing versions: {e}. Proceeding with registration.")
            return False

        if not versions:
            logger.info("No existing versions found — not a duplicate.")
            return False

        candidate_params = self.client.get_run(run_id).data.params

        for v in versions:
            try:
                existing_run = self.client.get_run(v.run_id)
            except Exception:
                continue  # run may have been deleted; skip it

            existing_params = existing_run.data.params
            existing_metric = existing_run.data.metrics.get(metric_key)

            params_match = all(
                str(candidate_params.get(k)) == existing_params.get(k) for k in candidate_params
            )
            metric_match = (
                existing_metric is not None and abs(existing_metric - best_score) <= metric_tol
            )

            if params_match and metric_match:
                logger.info(
                    f"Duplicate found: version {v.version} has identical params and "
                    f"{metric_key}={existing_metric:.4f} (candidate {metric_key}={best_score:.4f})."
                )
                return True

        logger.info(f"No duplicate found among {len(versions)} existing version(s).")
        return False

    def register_model(self, run_id):
        """Register the model to Unity Catalog."""
        logger.info(f"Registering model to {self.model_name}")
        try:
            model_details = mlflow.register_model(
                model_uri=f"runs:/{run_id}/{self.artifact_name}",
                name=self.model_name,
            )
            logger.info(f"Model registered: {model_details.name} version {model_details.version}")
            return model_details
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise

    def update_model_metadata(self, model_details, best_score):
        """Update model and version descriptions, set version tag."""
        logger.info(f"Updating metadata for {model_details.name} version {model_details.version}")
        try:
            self.client.update_registered_model(
                name=model_details.name,
                description="This model forecasts the demand of the next day based on product name, Destination_city.",
            )

            version_desc = (
                f"This model version has an R2 score metric of {round(best_score, 4) * 100}%. "
                "Follow the link to its training run for more details."
            )
            self.client.update_model_version(
                name=model_details.name,
                version=model_details.version,
                description=version_desc,
            )
            self.client.set_model_version_tag(
                name=model_details.name,
                version=model_details.version,
                key="val_r2",
                value=f"{round(best_score, 4)}",
            )
            logger.info("Model metadata updated successfully.")
        except Exception as e:
            logger.error(f"Failed to update model metadata: {e}")
            raise

    def run(self):
        """Run the full registration pipeline."""
        logger.info("Starting model registration pipeline...")
        try:
            self.setup_mlflow()
            run_id, best_score = self.find_best_run()

            if self.is_duplicate_model(run_id, best_score):
                logger.info("Identical model already registered — skipping registration.")
                return None

            model_details = self.register_model(run_id)
            self.update_model_metadata(model_details, best_score)
            logger.info("Pipeline completed successfully.")
            return model_details
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


# --- Usage example (uncomment to run standalone) ---
# catalog_name = "ct_oil_gas"
# gold_schema_name = "sc_gold"
# model_name = f"{catalog_name}.{gold_schema_name}.daily_oil_gas_demand_xg_model"
#
# RANDOM_STATE = 42
#
# current_user = 'abuhuzaifakurla@gmail.com'
# xp_name = "exp_daily_oil_gas_demand"
# run_name = 'daily_demand_xgb'
# artifact_name = "daily_oil_gas_demand_model"
#
# pipeline = ModelRegistrationPipeline(
#     model_name=model_name,
#     current_user=current_user,
#     experiment_name=xp_name,
#     run_name=run_name,
#     artifact_name=artifact_name,
# )
# pipeline.run()