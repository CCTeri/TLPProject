import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Constants
SHAP_SAMPLE_SIZE = 2000
SHAP_RANDOM_STATE = 42


class MultiModelComparer:
    """
    Multi-model comparison system for the product market share model.

    Compares 3 ML models using temporal validation:
    - Training: Jan 2024 - Nov 2024
    - Validation: Dec 2024 
    - Test: Jan 2025

    Jan 2024 ────── Nov 2024     Dec 2024     Jan 2025     Feb 2025
     │              │            │            │            │
     └─── TRAINING ─┘            │            │            │
                                 │            │            │
                            VALIDATION        │            │
                                              │            │
                                    Available Data   PREDICTION
                                    (for features)   (out-of-sample)

    Selects best model for final out-of-sample predictions.
    """

    def __init__(self, settings: dict, logger):
        self.settings = settings
        self.logger = logger

        # Results storage
        self.models = {}
        self.model_results = {}
        self.best_model_name = None
        self.best_model = None
        self.data_splits = {}
        self.features = None
        self.scaler = None

    def prepare_temporal_splits(self, df_scaled: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Construct time-ordered train, validation, and test partitions.

        Args:
            df_scaled: Scaled DataFrame including features and target.
        """
        self.logger.info("[MultiModel] Preparing temporal data splits...")

        df = df_scaled.copy()
        if df['date'].dtype.name == 'period[M]':
            df['date'] = df['date'].dt.to_timestamp()

        # Define split boundaries
        train_end = pd.Timestamp('2024-11-30')
        val_end = pd.Timestamp('2024-12-31')

        # Split the data into train, valid, and test
        splits = {
            'train': df[df['date'] <= train_end].copy(),
            'validation': df[(df['date'] > train_end) & (df['date'] <= val_end)].copy(),
            'test': df[df['date'] > val_end].copy()
        }

        # Log split information
        for split_name, split_data in splits.items():
            if not split_data.empty:
                date_range = f"{split_data['date'].min().strftime('%Y-%m')} to {split_data['date'].max().strftime('%Y-%m')}"
                self.logger.info(f"  {split_name.capitalize()}: {len(split_data)} rows ({date_range})")

        self.data_splits = splits
        return splits

    def get_model_features(self, df: pd.DataFrame) -> List[str]:
        """
        Retrieve the full feature set defined in FeatureEngineer and
        validate it against the actual columns in the modeling dataset.

        Args:
            df: DataFrame containing the modeling data.
        """
        # Bring the feature choices from the feature class
        from src.feature import FeatureEngineer
        feature_engineer = FeatureEngineer(self.settings, self.logger)
        selected_features = feature_engineer.get_modeling_features()

        # Validate features exist in data
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = [f for f in selected_features if f not in df.columns]

        if missing_features:
            self.logger.warning(f"[MultiModel] Missing features: {missing_features}")

        self.logger.info(f"[MultiModel] Using {len(available_features)} features for modeling")
        self.features = available_features
        return available_features

    def train_and_compare_models(self, df_scaled: pd.DataFrame, scaler, target: str = 'weight_share'):
        """
        Train multiple model candidates and compare their performance.

        This method prepares the input data, trains three predefined models
        (LightGBM, Random Forest, and Linear Regression), evaluates their performance,
        and selects the best model based on predefined metrics.

        Parameters:
            df_scaled (pd.DataFrame): Scaled input DataFrame including features and target.
            scaler: Scaler object used for preprocessing (stored for potential inverse transforms).
            target (str): The name of the target column to predict. Defaults to 'weight_share'.

        Returns:
            self: The instance with updated `models` and `model_results` attributes,
                  including evaluation metrics and feature importance for each model.
        """
        self.logger.info("[MultiModel] Starting model comparison...")
        self.scaler = scaler

        # Prepare data splits
        splits = self.prepare_temporal_splits(df_scaled)
        features = self.get_model_features(df_scaled)

        # Prepare datasets
        X_train = splits['train'][features].fillna(0)
        y_train = splits['train'][target]
        X_val = splits['validation'][features].fillna(0) if not splits['validation'].empty else None
        y_val = splits['validation'][target] if not splits['validation'].empty else None

        # Define models to train
        models_config = {
            'lightgbm': {
                'name': 'LightGBM',
                'class': lgb.LGBMRegressor,
                'params': self.settings['lightgbm_params'],
                'needs_validation': True
            },
            'random_forest': {
                'name': 'Random Forest',
                'class': RandomForestRegressor,
                'params': self.settings['rf_params'],
                'needs_validation': False
            },
            'linear_regression': {
                'name': 'Linear Regression',
                'class': LinearRegression,
                'params': self.settings['lr_params'],
                'needs_validation': False
            }
        }

        # Train each model
        for model_key, config in models_config.items():
            self.logger.info(f"[{config['name']}] Training")

            # Create model
            model = config['class'](**config['params'])

            # Train with or without validation
            if config['needs_validation'] and X_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric=self.settings['eval_metric'],
                )
            else:
                model.fit(X_train, y_train)

            # Store model and evaluate
            self.models[model_key] = model
            metrics = self._evaluate_model(model, model_key, splits, features, target)

            self.model_results[model_key] = {
                'name': config['name'],
                'metrics': metrics,
                'feature_importance': self._get_feature_importance(model, features)
            }

            self.logger.info(f"[{config['name']}] Training completed")

        # Select best model and log results
        self._select_best_model()
        self._log_model_comparison()

        # SHAP importance for the best model (on validation split)
        val_split = self.data_splits.get("validation")

        if val_split is not None and not val_split.empty:
            X_val_shap = val_split[self.features].fillna(0)
            X_sample = X_val_shap.sample(min(SHAP_SAMPLE_SIZE, len(X_val_shap)), random_state=SHAP_RANDOM_STATE)

            try:
                shap_importance = self._get_feature_importance(
                    self.best_model,
                    self.features,
                    X_sample=X_sample,  # <- triggers SHAP mode
                )

                # overwrite default importance with SHAP for the best model
                self.model_results[self.best_model_name]["feature_importance"] = shap_importance
                self.model_results[self.best_model_name]["shap_importance"] = shap_importance

                self.logger.info("SHAP importance computed for the best model.")
                self.logger.info(f"Top SHAP features:\n{shap_importance.head()}")
            except Exception as e:
                self.logger.warning(f"Could not compute SHAP importance: {e}")
        else:
            self.logger.info("No validation split available for SHAP computation.")

        return self

    def _evaluate_model(self, model, model_key: str, splits: Dict[str, pd.DataFrame],
                        features: List[str], target: str) -> Dict[str, Dict[str, float]]:
        """
        Compute performance statistics of a model on a specific data split.

        Produces:
            - RMSE
            - MAE
            - R²
            - MAPE (with safeguards against zero division)
            - sample size

        Metrics are stored and surfaced for reporting and comparison.
        """
        metrics = {}

        for split_name, split_data in splits.items():
            if split_data.empty:
                continue

            X = split_data[features].fillna(0)
            y_true = split_data[target]
            y_pred = model.predict(X)

            metrics[split_name] = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100,
                'samples': len(y_true)
            }

        return metrics

    def _get_feature_importance(self, model, features: List[str], X_sample: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get feature importance of the best model using SHAP values.

        Args:
            model: Trained model instance.
            features: List of feature names used in training.
        """
        if X_sample is not None:
            # Choose SHAP explainer based on model type
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, "coef_"):
                explainer = shap.LinearExplainer(model, X_sample)
            else:
                explainer = shap.KernelExplainer(model.predict, X_sample)

            shap_values = explainer.shap_values(X_sample)

            # Some SHAP backends return list — use first element
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            return (
                pd.DataFrame({
                    "feature": features,
                    "importance": mean_abs_shap,
                })
                .sort_values("importance", ascending=False)
            )

        # ---- fallback: normal importance when X_sample is None ----
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
        else:
            importance = np.zeros(len(features))

        return (
            pd.DataFrame({
                "feature": features,
                "importance": importance,
            })
            .sort_values("importance", ascending=False)
        )

    def _select_best_model(self):
        """
        Select the best model based on validation RMSE (or test RMSE if no validation).
        """
        # Choose metric for selection (prefer validation, fallback to test, then train)
        selection_criteria = []
        for model_key, results in self.model_results.items():
            metrics = results['metrics']

            if 'validation' in metrics:
                score = metrics['validation']['rmse']
                criteria = 'validation_rmse'
            elif 'test' in metrics:
                score = metrics['test']['rmse']
                criteria = 'test_rmse'
            else:
                score = metrics['train']['rmse']
                criteria = 'train_rmse'

            selection_criteria.append({
                'model': model_key,
                'name': results['name'],
                'score': score,
                'criteria': criteria
            })

        # Select model with lowest RMSE
        best_model_info = min(selection_criteria, key=lambda x: x['score'])
        self.best_model_name = best_model_info['model']
        self.best_model = self.models[self.best_model_name]

        self.logger.info(f"[MultiModel] Best model selected: {best_model_info['name']} "
                         f"({best_model_info['criteria']}: {best_model_info['score']:.4f})")

    def _log_model_comparison(self):
        """
        Log comprehensive model comparison results.
        """
        self.logger.info("MULTI-MODEL COMPARISON RESULTS")

        for model_key, results in self.model_results.items():
            self.logger.info(f"\n {results['name'].upper()}:")

            for split_name, metrics in results['metrics'].items():
                self.logger.info(f"{split_name.capitalize()} ({metrics['samples']} samples):")
                self.logger.info(f"RMSE:{metrics['rmse']:.4f}")
                self.logger.info(f"MAE:{metrics['mae']:.4f}")
                self.logger.info(f"R²:{metrics['r2']:.4f}")
                self.logger.info(f"MAPE:{metrics['mape']:.2f}%")

        # Highlight best model
        if self.best_model_name:
            best_results = self.model_results[self.best_model_name]
            best_name = best_results['name']
            self.logger.info(f"\nSELECTED FOR PRODUCTION: {best_name}")

            fi = best_results.get('feature_importance')

            self.logger.info(f"\nTop 5 Features ({best_name}):")
            if isinstance(fi, pd.DataFrame) and not fi.empty:
                top_features = fi.head(5)
                for _, row in top_features.iterrows():
                    self.logger.info(f"  {row['feature']}: {row['importance']:.2f}")
            else:
                self.logger.info("  (No feature importance available)")

    def predict_future(self, df_route: pd.DataFrame) -> pd.DataFrame:
        """
        Generate out-of-sample predictions using the best model.

        Args:
            df_route: Original route data (unscaled)
            target_date: Target prediction date

        Returns:
            DataFrame with predictions including lag data for Power BI
        """
        target_date = self.settings['prediction_date']

        self.logger.info(
            f"[MultiModel] Generating predictions for {target_date} using {self.model_results[self.best_model_name]['name']}")

        # Copy and find the last historic period
        last_period = df_route['date'].max()

        # Take only the last-period rows (one row per product×O×D with all lag/time features)
        last_feat = df_route[df_route['date'] == last_period].copy()

        # Build future frame by reusing those last-period feature rows
        df_fut = last_feat.copy()

        # Parse target date from settings and update temporal features
        fut_month = pd.to_datetime(target_date)
        df_fut['date'] = fut_month.to_period('M')
        df_fut['month'] = fut_month.month
        df_fut['quarter'] = fut_month.quarter

        # "t" is a simple month index from the earliest year in history
        min_year = df_route['date'].dt.year.min()
        df_fut['t'] = (fut_month.year - min_year) * 12 + fut_month.month

        # Run the model using the stored features from training
        best_model = self.models[self.best_model_name]
        df_fut['pred_share'] = best_model.predict(df_fut[self.features])

        # Pick the winning product per route by highest pred_share
        top = (
            df_fut
            .sort_values(['origin_city', 'destination_city', 'pred_share'], ascending=[True, True, False])
            .groupby(['origin_city', 'destination_city'], as_index=False)
            .first()  # first() after sorting gives the max-share row
        )

        return top

    def get_model_comparison_summary(self) -> Dict:
        """
        Get comprehensive comparison summary for reporting.

        Returns:
            Dictionary summarizing model comparison results.
        """
        summary = {
            'comparison_strategy': 'Temporal Validation (Train: Jan-Nov 2024, Val: Dec 2024, Test: Jan 2025)',
            'models_compared': len(self.model_results),
            'best_model': {
                'name': self.model_results[self.best_model_name]['name'],
                'key': self.best_model_name
            } if self.best_model_name else None,
            'detailed_results': self.model_results,
            'features_count': len(self.features) if self.features else 0
        }

        return summary
