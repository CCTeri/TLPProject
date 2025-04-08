import yaml
import os
import pandas as pd
from src.logger import init_logger
from src.reader import Reader
from src.writer import Writer


def run_project():
    # Load settings from settings.yml file and environment variables
    with open('settings.yml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize logger for a selected period
    logger = init_logger(settings)
    logger.info(f'[>] Running TLP Project: Niche Market Research for Cargo')

    # Read data from GCS
    bucket_name = 'tlp_project_demo'
    gcs_input_path = 'Input/marketdata500.csv'
    # df_wacd = Reader(settings, logger).read_data(bucket_name, gcs_input_path)
    df_wacd = Reader(settings, logger).read_data()


    # # Write output directory in GCP
    # gcs_output_path = 'Output/top_30_marketdata.csv'  # Define the output path for the CSV
    # Writer(settings, logger).write_data(df_wacd, bucket_name, gcs_output_path)  # Call the function from writer.py
    # logger.info(f"Data saved to {gcs_output_path}")

    # Complete the job
    logger.info('[V] Finished')


if __name__ == "__main__":
    run_project()
