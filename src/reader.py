import pandas as pd
from io import StringIO
from google.cloud import storage
from datetime import datetime
from dateutil.relativedelta import relativedelta


class Reader:
    """
    Reader for market data, supporting both local files and GCS.

    Args:
        settings (dict): Project settings containing 'period_month', 'project_input', etc.
        logger (logging.Logger): Logger object for informational and error messages.
    """

    def __init__(self, settings: dict, logger):
        self.settings = settings
        self.logger = logger

        # Parse the reporting period
        period = datetime.strptime(settings['period_month'], '%Y%m')
        self.period_month = period.strftime('%Y%m')
        self.start_date = period.strftime('%Y-%m-%d')
        self.end_date = (period + relativedelta(months=1) - relativedelta(days=1)).strftime('%Y-%m-%d')

    def read_data(self, bucket_name: str, file_name: str) -> pd.DataFrame:
        """
        Read data from GCS.

        Args:
            bucket_name (str): Name of the GCS bucket.
            file_name (str): Path to the file within the bucket.

        Returns:
            pd.DataFrame: Loaded data frame.
        """
        self.logger.info(f"[>] Reading data from GCS: {bucket_name}/{file_name}")
        return self._read_from_gcs(bucket_name, file_name)

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
        client = storage.Client()
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(file_name)
            data = blob.download_as_text()
            return pd.read_csv(StringIO(data))
        except Exception as e:
            self.logger.error(f"Error reading file {file_name} from bucket {bucket_name}: {e}")
            return pd.DataFrame()