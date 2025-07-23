import yaml
import os
import pandas as pd
from src.logger import init_logger
from src.reader import Reader
from src.writer import Writer
from src.processor import DataProcessor
from src.modeler import Modeler

def run_project():
    """
    Executes the TLP Project pipeline:
      1. Load settings
      2. Initialize logger
      3. Read data (GCS or local)
      4. Process data
      5. Write output
    """
    # Load settings file
    settings_path = os.getenv('SETTINGS_PATH', 'settings.yml')
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)

    # Initialize logger
    logger = init_logger(settings)
    logger.info('Starting TLP Project: Niche Market Research for Cargo')

    # Data source configuration
    bucket = settings.get('gcs_bucket', 'tlp_project_demo')
    gcs_input = settings.get('gcs_input_path', 'Input/market_20240101-20250101_product.csv')

    # Read data
    df_wacd = Reader(settings, logger).read_data(bucket, gcs_input)

    # Process data
    logger.info('Processing data')
    df_route = DataProcessor(settings, logger).process_data(df_wacd)

    # Train the model on historical route×product data
    modeler = Modeler(settings, logger)

    # TODO: seperate main file for train and predict. then predict only goes to GCS
    modeler.train(df_route)

    # Forecast for Feb 2025 (Out of sample)
    top_feb25 = modeler.predict_future(df_route, date='2025-02')

    # Output configuration
    output_bucket = settings.get('gcs_bucket', bucket)
    output_path = settings.get('gcs_output_path', 'Output/predicted_product_share.csv')

    # Write results
    logger.info(f'Writing processed data to gs://{output_bucket}/{output_path}')
    Writer(settings, logger).write_data(top_feb25, output_bucket, output_path)

    logger.info('TLP Project completed successfully')

if __name__ == '__main__':
    run_project()