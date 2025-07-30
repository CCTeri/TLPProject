import pandas as pd
from google.cloud import storage
import io
import os


class Writer(object):
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

    def write_data(self, df, bucket_name, output_path):
        """
        Save the DataFrame to GCS as a CSV.

        :param df: The DataFrame to save
        :param bucket_name: The GCS bucket name
        :param output_path: The GCS path where the file will be saved
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

    def save_output(self, df):
        """
        Save the output locally for Power BI.

        :param df: The DataFrame to save
        """
        # Create directory if it doesn't exist
        os.makedirs('data/output', exist_ok=True)

        # Save to local CSV
        output_file = 'data/output/prediction_202502.csv'
        df.to_csv(output_file, index=False)

        self.logger.info(f"The prediction is saved locally to {output_file}")

        return output_file

