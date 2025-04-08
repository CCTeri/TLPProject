import pandas as pd
# import numpy as np
# import zipfile
# from typing import Tuple
from datetime import datetime
# from sqlalchemy import create_engine, text
# from sqlalchemy.engine import URL
# from itertools import product
from dateutil.relativedelta import relativedelta
import pandas as pd
import gcsfs

class Reader(object):
    """
    Class to read in the input data

    Args:
        settings (dict): Project settings
        logger (logging.Logger): Logger object
    """

    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger

        # Set the Year Month to datetime
        self.period_month = datetime.strptime(self.settings['period_month'], '%Y%m').strftime('%Y%m')

        # Start and End date are used to determine which Market data to read
        self.start_date = datetime.strptime(self.settings['period_month'], '%Y%m').strftime('%Y-%m-%d')
        self.end_date = (datetime.strptime(self.settings['period_month'], '%Y%m') + relativedelta(months = 1) - relativedelta(days = 1)).strftime('%Y-%m-%d')

        pass

    def read_data(self):
        """
        Read the input data

        Args:
        Returns:
            df_wacd (pandas.DataFrame): Market data from WACD for every carrier on city-city level per period (month)
        """
        self.logger.info(f'[>] Reading data')

        df_wacd = self._read_WACD_local_data()
        # df_wacd = self.read_data_from_gcs(bucket_name, gcs_input_path)

        return df_wacd

    def _read_WACD_local_data(self):
        """
        Read a local WACD data

        :return:
            pandas.Dataframe: Market data for a year
        """
        self.logger.info(f'\t\t[>] Reading WACD file')
        path = self.settings['project_input'] + self.settings['WACD_Local']

        try:
            df = pd.read_csv(path, sep="\t", index_col=False)
            return df

        except Exception as e:
            print(f"Error reading market data: {e}")
            return None

    def read_data_from_gcs(self, bucket_name, file_name):
        """
        Reads a CSV file from Google Cloud Storage (GCS)
        :param gcs_path: The GCS path to the input file
        :return: DataFrame
        """
        client = storage.Client()  # Create a GCS client
        try:
            bucket = client.get_bucket(bucket_name)  # Get the GCS bucket
            blob = bucket.blob(file_name)  # Get the file within the bucket
            data = blob.download_as_text()  # Download file as text
            df = pd.read_csv(pd.compat.StringIO(data))  # Convert text to pandas DataFrame
            return df
        except Exception as e:
            self.logger.error(f"Error reading file {file_name} from GCS bucket {bucket_name}: {e}")
            return None