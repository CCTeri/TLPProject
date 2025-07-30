import yaml
import os
import pandas as pd
from src.logger import init_logger
from src.reader import Reader
from src.writer import Writer
from src.processor import DataProcessor
from src.feature import FeatureEngineer
from src.scaler import SimpleScaler, DomainSpecificScaler
from src.modeler import MultiModelComparer  # Use the simple version


def run_project():
    """
    Simple TLP Project pipeline:
      1. Load settings
      2. Initialize logger
      3. Read data (GCS or local)
      4. Process data with feature engineering
      5. Scale features
      6. Compare 3 models and pick best one
      7. Generate Feb 2025 predictions
      8. Save output locally
    """
    # Load settings file
    settings_path = os.getenv('SETTINGS_PATH', 'settings.yml')
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)

    # Initialize logger
    logger = init_logger(settings)
    logger.info('Starting TLP Project: Product Demand Prediction for Cargo')

    # Data source configuration
    bucket = settings.get('gcs_bucket', 'tlp_project_demo')
    gcs_input = settings.get('gcs_input_path', 'Input/market_20240101-20250101_product.csv')

    # Read data
    logger.info('Reading data')
    df_wacd = Reader(settings, logger).read_data(bucket, gcs_input)

    # Process data
    logger.info('Processing data')
    df_route = DataProcessor(settings, logger).process_data(df_wacd)

    # Engineer features
    logger.info('Engineering features')
    df_features = FeatureEngineer(settings, logger).build_features(df_route)

    # Scale the data
    logger.info('Scaling features')
    scaler_type = settings.get('scaling_method', 'simple')
    if scaler_type == 'domain':
        scaler = DomainSpecificScaler(settings, logger)
    else:
        scaler = SimpleScaler(settings, logger)

    df_scaled = scaler.fit_transform(df_features)

    # Train and compare models
    logger.info('Comparing models and selecting best one')
    model_comparer = MultiModelComparer(settings, logger)
    best_model = model_comparer.train_and_compare_models(df_scaled, scaler)

    # Forecast using the best model (df_route for feature engineering, not df_scaled)
    logger.info('Generating predictions for target month')
    feb_predictions = best_model.predict_future(df_features)

    # Save output locally
    logger.info('Saving predictions locally')
    writer = Writer(settings, logger)
    local_file = writer.save_output(feb_predictions)

    # Summary
    logger.info('TLP Project completed successfully')
    logger.info(f"Best model: {model_comparer.best_model_name}")
    logger.info(f"Predictions generated: {len(feb_predictions)} routes")
    logger.info(f"Output saved to: {local_file}")

    return feb_predictions


if __name__ == '__main__':
    predictions = run_project()
