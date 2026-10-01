import logging
import numpy as np

import xgboost as xgb
import mlflow
import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Configure simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DemandModelTrainer:
    """Train an XGBoost demand forecasting model with MLflow tracking."""

    def __init__(self, spark, catalog, schema, feature_table, labels_table,
                 eval_split_table, experiment_path, random_state=42):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.feature_table = feature_table
        self.labels_table = labels_table
        self.eval_split_table = eval_split_table
        self.experiment_path = experiment_path
        self.random_state = random_state
        self.fe = FeatureEngineeringClient()
        self.cat_cols = ["product_name", "destination_city"]

    def _full_name(self, table):
        return f"{self.catalog}.{self.schema}.{table}"

    def setup_mlflow(self):
        """Configure MLflow autolog, registry URI, and experiment."""
        logger.info("Setting up MLflow...")
        try:
            mlflow.xgboost.autolog(log_models=False)
            mlflow.set_registry_uri("databricks-uc")
            mlflow.set_experiment(self.experiment_path)
            logger.info(f"MLflow experiment set to: {self.experiment_path}")
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
            raise

    def load_training_data(self):
        """Load labels and split tables, build training set via Feature Store."""
        logger.info("Loading training data...")
        try:
            labels_df = self.spark.table(self._full_name(self.labels_table)).select(
                F.col("total_demand"), "p_key", "transaction_date", "split"
            )
            train_eval_df = labels_df.filter("split == 'train'").select(
                "p_key", "transaction_date", "total_demand"
            )

            feature_lookups = [
                FeatureLookup(
                    table_name=self._full_name(self.feature_table),
                    lookup_key="p_key",
                    timestamp_lookup_key="transaction_date",
                )
            ]
            training_set = self.fe.create_training_set(
                df=train_eval_df,
                feature_lookups=feature_lookups,
                label="total_demand",
                exclude_columns=["p_key"],
            )
            training_pdf = training_set.load_df().toPandas().dropna()
            logger.info(f"Training data loaded: {training_pdf.shape[0]} rows, {training_pdf.shape[1]} columns")
            return training_set, training_pdf
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise

    def prepare_splits(self, training_pdf, train_ratio=0.8):
        """Split into X/y and create time-based train/test sets."""
        logger.info(f"Preparing train/test split with ratio={train_ratio}")
        try:
            X = training_pdf.drop("total_demand", axis=1)
            y = training_pdf["total_demand"]

            sorted_idx = X["transaction_date"].sort_values().index
            X_sorted = X.loc[sorted_idx]
            y_sorted = y.loc[sorted_idx]
            split_point = int(len(X_sorted) * train_ratio)

            X_train = X_sorted.iloc[:split_point].drop(columns=["transaction_date"])
            X_test = X_sorted.iloc[split_point:].drop(columns=["transaction_date"])
            y_train = y_sorted.iloc[:split_point]
            y_test = y_sorted.iloc[split_point:]

            logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Data split failed: {e}")
            raise

    def _to_str(self, df):
        """Convert categorical columns to string type."""
        df = df.copy()
        df[self.cat_cols] = df[self.cat_cols].astype(str)
        return df

    def train(self, X_train, X_test, y_train, y_test, training_set):
        """Build pipeline, train model, and log to MLflow."""
        logger.info("Starting model training with MLflow...")
        try:
            X_train_s = self._to_str(X_train)
            X_test_s = self._to_str(X_test)

            pre = ColumnTransformer(
                [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), self.cat_cols)],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas")

            model = Pipeline([
                ("pre", pre),
                ("xgb", xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=self.random_state,
                )),
            ])

            with mlflow.start_run(run_name="daily_demand_xgb") as run:
                model.fit(X_train_s, y_train)
                mlflow.log_params(model["xgb"].get_params())

                y_pred = model.predict(X_test_s)
                val_mae = mean_absolute_error(y_test, y_pred)
                val_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                val_r2 = r2_score(y_test, y_pred)
                mlflow.log_metrics({"val_mae": val_mae, "val_rmse": val_rmse, "val_r2": val_r2})

                self.fe.log_model(
                    model=model,
                    artifact_path="daily_oil_gas_demand_model",
                    flavor=mlflow.sklearn,
                    training_set=training_set,
                    registered_model_name=None,
                    input_example=X_train_s.iloc[[0]],
                    signature=infer_signature(X_train_s, y_pred),
                )

                logger.info(f"Training complete. MAE={val_mae:.2f}, RMSE={val_rmse:.2f}, R2={val_r2:.4f}, run_id={run.info.run_id}")
                return run.info.run_id
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    def run(self):
        """Run the full training pipeline."""
        logger.info("Starting demand model training pipeline...")
        try:
            self.setup_mlflow()
            training_set, training_pdf = self.load_training_data()
            X_train, X_test, y_train, y_test = self.prepare_splits(training_pdf)
            run_id = self.train(X_train, X_test, y_train, y_test, training_set)
            logger.info("Pipeline completed successfully.")
            return run_id
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


# --- Usage example (uncomment to run standalone) ---
# catalog_name = "ct_oil_gas"
# gold_schema_name = "sc_gold"
# eval_split_table_name = "demand_eval_split"
# feature_table_name = "daily_demand_features"
# demand_labels_table_name = "daily_demand_labels"
# experiment_path = "/Users/abuhuzaifakurla@gmail.com/exp_daily_oil_gas_demand"
#
# RANDOM_STATE = 42
#
# trainer = DemandModelTrainer(
#     spark=spark,
#     catalog=catalog_name,
#     schema=gold_schema_name,
#     feature_table=feature_table_name,
#     labels_table=demand_labels_table_name,
#     eval_split_table=eval_split_table_name,
#     experiment_path=experiment_path,
#     random_state=RANDOM_STATE,
# )
# run_id = trainer.run()