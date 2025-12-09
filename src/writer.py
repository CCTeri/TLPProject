import pandas as pd
from google.cloud import storage
import os


class Writer:
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

    def write_data(self, df, bucket_name, output_path):
        """
        Save the DataFrame to GCS as a CSV.

        Args:
            df: The DataFrame to save
            bucket_name: The GCS bucket name
            output_path: The GCS path where the file will be saved
        """
        # Create a GCS client
        client = storage.Client()

        # Get the bucket
        bucket = client.get_bucket(bucket_name)

        # Get the blob (file) in GCS
        blob = bucket.blob(output_path)

        # Save to CSV in memory and upload to GCS
        with blob.open("w") as f:
            df.to_csv(f, index=False)
        self.logger.info(f"The prediction is saved to {output_path} in GCS")

    def save_output(self, df: pd.DataFrame, prediction_date: str = None) -> str:
        """
        Save the output locally for Power BI.

        Args:
            df: The DataFrame to save
            prediction_date: Optional prediction date string (e.g., "2025-02") for filename

        Returns:
            str: Path to the saved output file
        """
        # Create directory if it doesn't exist
        os.makedirs('data/output', exist_ok=True)

        # Save to local CSV
        output_file = 'data/output/prediction_202502.csv'
        df.to_csv(output_file, index=False)
        self.logger.info(f"Prediction saved locally to {output_file}")

        return output_file

