import pandas as pd
import os
from io import StringIO
from google.cloud import storage
from datetime import datetime
from dateutil.relativedelta import relativedelta


class Reader:
    """
    Reader for market data, supporting both local files and GCS.

    Args:
        settings (dict): Project settings containing 'data_source', 'project_input', etc.
        logger (logging.Logger): Logger object for informational and error messages.
    """

    def __init__(self, settings: dict, logger):
        self.settings = settings
        self.logger = logger

        # Parse the reporting period
        # period = datetime.strptime(settings['period_month'], '%Y%m')
        # self.period_month = period.strftime('%Y%m')
        # self.start_date = period.strftime('%Y-%m-%d')
        # self.end_date = (period + relativedelta(months=1) - relativedelta(days=1)).strftime('%Y-%m-%d')

    def read_data(self, bucket_name: str = None, file_name: str = None) -> pd.DataFrame:
        """
        Read data from GCS or local file based on settings configuration.

        Args:
            bucket_name (str): Name of the GCS bucket (required if data_source is "gcs").
            file_name (str): Path to the file within the bucket (required if data_source is "gcs").

        Returns:
            pd.DataFrame: Loaded data frame.
        """
        data_source = self.settings.get('data_source', 'gcs').lower()
        
        if data_source == 'local':
            return self.read_local()
        elif data_source == 'gcs':
            if not bucket_name or not file_name:
                self.logger.error("GCS bucket_name and file_name are required when data_source is 'gcs'")
                return pd.DataFrame()
            return self._read_from_gcs(bucket_name, file_name)
        else:
            self.logger.error(f"Invalid data_source '{data_source}'. Must be 'gcs' or 'local'")
            return pd.DataFrame()

    def read_local(self) -> pd.DataFrame:
        """
        Read the local WACD file defined in settings.

        Returns:
            pd.DataFrame: Loaded data frame.
        """
        path = f"{self.settings['project_input']}/{self.settings['WACD_Local']}"
        self.logger.info(f"[>] Reading local file: {path}")
        try:
            return pd.read_csv(path, sep="\t")
        except Exception as e:
            self.logger.error(f"Error reading local file {path}: {e}")
            return pd.DataFrame()

    def _read_from_gcs(self, bucket_name: str, file_name: str) -> pd.DataFrame:
        """
        Internal: Download and parse a CSV from GCS.
        """
        # Set up GCS credentials if key file exists
        key_path = os.path.join('key', 'key_cloudstorage.json')
        if os.path.exists(key_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath(key_path)
        
        # Authenticate with Google Cloud Storage
        try:
            client = storage.Client()
            # Reference the bucket that we want to use
            bucket = client.bucket(bucket_name)
            # Reference the specific file within that bucket
            blob = bucket.blob(file_name)
            # Download the blob contents from GCS into memory
            data = blob.download_as_text()
            # Parse that text with pandas using StringIO and returns DF in one shot
            return pd.read_csv(StringIO(data), sep='\t')
        except Exception as e:
            self.logger.error(f"Error reading file {file_name} from bucket {bucket_name}: {e}")
            return pd.DataFrame()