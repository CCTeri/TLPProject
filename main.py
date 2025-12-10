import yaml
import os
from typing import Optional
import pandas as pd
from src.logger import init_logger
from src.reader import Reader
from src.writer import Writer
from src.processor import DataProcessor
from src.feature import FeatureEngineer
from src.scaler import SimpleScaler, DomainSpecificScaler
from src.modeler import MultiModelComparer


def run_project() -> Optional[pd.DataFrame]:
    """
    Execute the complete TLP Project pipeline for product demand prediction.

    This function orchestrates the entire machine learning pipeline:
    1. Load configuration settings from YAML file
    2. Initialize logging system
    3. Read market data from GCS or local file system
    4. Process and clean raw data
    5. Engineer features for modeling
    6. Scale features for model training
    7. Train and compare multiple models (LightGBM, Random Forest, Linear Regression)
    8. Select best model based on validation performance
    9. Generate predictions for target prediction date
    10. Save predictions to GCS 

    Returns:
        DataFrame containing predictions with columns for origin, destination,
        product, and predicted market share. Returns None if pipeline fails.

    Raises:
        FileNotFoundError: If settings file or data file cannot be found
        ValueError: If required settings are missing or data is invalid
    """
    # Load settings file
    settings_path = os.getenv('SETTINGS_PATH', 'settings.yml')
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)

    # Initialize logger
    logger = init_logger(settings)
    logger.info('Starting TLP Project: Product Demand Prediction for Cargo')

    # Data source configuration
    data_source = settings.get('data_source', 'gcs').lower()
    bucket = settings.get('gcs_bucket', 'tlp_project_demo')
    gcs_input = settings.get('gcs_input_path', 'Input/market_20240101-20250101_product.csv')

    # Read data
    logger.info(f'Reading data from {data_source.upper()}')
    reader = Reader(settings, logger)
    if data_source == 'local':
        df_market = reader.read_data()
    else:
        df_market = reader.read_data(bucket, gcs_input)

    # Validate data was loaded successfully
    if df_market.empty:
        logger.error("No data loaded. Please check data source configuration.")
        return None
    logger.info(f"Loaded {len(df_market)} rows of data")

    # Process data
    logger.info('Processing data')
    df_route = DataProcessor(settings, logger).process_data(df_market)

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
    model_comparer.train_and_compare_models(df_scaled, scaler)

    # Forecast using the best model
    logger.info('Generating predictions for target month')
    predictions = model_comparer.predict_future(df_features)

    # Save output
    logger.info('Saving predictions')
    writer = Writer(settings, logger)
    
    # Save to GCS
    gcs_bucket = settings.get('gcs_bucket')
    gcs_output_path = settings.get('gcs_output_path')
    
    if gcs_bucket and gcs_output_path:
        try:
            writer.write_data(predictions, gcs_bucket, gcs_output_path)
        except Exception as e:
            logger.warning(f"Failed to save to GCS (continuing with local save): {e}")

    # Complete the generation
    logger.info('TLP Project completed successfully')
    
    return predictions


if __name__ == '__main__':
    predictions = run_project()
