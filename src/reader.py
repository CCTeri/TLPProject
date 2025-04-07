import pandas as pd
# import numpy as np
# import zipfile
# from typing import Tuple
from datetime import datetime
# from sqlalchemy import create_engine, text
# from sqlalchemy.engine import URL
# from itertools import product
from dateutil.relativedelta import relativedelta

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
            df = pd.read_csv(path, sep="_", index_col=False)
            return df

        except Exception as e:
            print(f"Error reading market data: {e}")
            return None

    # def _read_market_data(self):
    #     """
    #     Read the WACD market data files on city - city level with three carrier groups (AF, KL and OTH).
    #
    #     Args:
    #
    #     Returns:
    #         pandas.DataFrame: Market data from WACD for every carrier on city-city level per period (month)
    #     """
    #     self.logger.info(f'[>] Reading WACD market data from SQL')
    #
    #     # TODO: Add yield and product types
    #     engine = self._init_engine()
    #     query = f"""
    #     SELECT
    #         OriginCity AS DepartureCity,
    #         DestinationCity AS ArrivalCity,
    #         Period AS YearMonth,
    #         Carrier,
    #         ChargeableWeight
    #     FROM
    #         RM_History.dbo.tblAGGWACD
    #     WHERE
    #         LEN(OriginCity) = 3 AND
    #         LEN(DestinationCity) = 3 AND
    #         ChargeableWeight > 0 AND
    #         Period >= '{self.start_date}' AND
    #         Period <= '{self.end_date}'
    #     """
    #
    #     with engine.connect() as conn:
    #         df = pd.read_sql_query(text(query), conn)
    #     df['YearMonth'] = pd.to_datetime(df['YearMonth']).dt.strftime('%Y%m').astype(int)
    #     # Chargeable weight is in kilos
    #     df['ChargeableWeight'] = df['ChargeableWeight'].astype(int)
    #
    #     # Only keep O&Ds which are within the top cities we defined
    #     df['OD'] = df['DepartureCity'] + '-' + df['ArrivalCity']
    #
    #     return df
