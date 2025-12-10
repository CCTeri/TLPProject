from google.cloud import storage


class Writer:
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

    def write_data(self, df, bucket_name, output_path):
        """
        Save the DataFrame to GCS as a CSV

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


