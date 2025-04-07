import io
import pandas as pd
from google.cloud import storage


class Reader:
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

    def read_data_from_gcs(self, bucket_name, file_name):
        """
        Reads a CSV file from Google Cloud Storage (GCS)
        :param bucket_name: The name of the GCS bucket
        :param file_name: The path to the file inside the GCS bucket
        :return: DataFrame or None if there's an error
        """
        client = storage.Client()  # Create a GCS client

        try:
            # Get the GCS bucket
            bucket = client.get_bucket(bucket_name)
            # Get the file blob from the bucket
            blob = bucket.blob(file_name)
            # Download file content as text
            data = blob.download_as_text()
            # Read the text data into a pandas DataFrame
            df = pd.read_csv(io.StringIO(data))  # Use StringIO to read the text as file-like object
            return df
        except Exception as e:
            # Log the error if something goes wrong
            self.logger.error(f"Error reading file {file_name} from GCS bucket {bucket_name}: {e}")
            return None
