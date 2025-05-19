import pandas as pd
from google.cloud import storage
import io

class Writer(object):
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

    def write_data(self, df, bucket_name, output_path):
        """
        Save the top 30 rows of the DataFrame to GCS as a CSV.

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

        # Save the top 30 rows to CSV in memory and upload to GCS
        with blob.open("w") as f:
            df.to_csv(f, index=False)
        self.logger.info(f"The prediction is saved to {output_path} in GCS")

        # TODO: add another output that shows rising market (=> can be shown in the dashboard?)
        # TODO: Connect Power BI with the output of GCS
        # Q. How can i see the output in the GCS in a table format?

