from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """
    Process the data

    :arg
        settings (dict): Project settings
        logger (logging.Logger): Logger object
    """

    def __init__(self, settings: dict, logger):
        self.settings = settings
        self.logger = logger

        pass

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all functions for data processing
        """
        self.logger.info('[>] Begin data processing')
        df = self._clean(df)

        return df

    def _clean(self, df) -> pd.DataFrame:
        """
        Clean raw data: drop NAs, correct dtypes, etc.
        """
        self.logger.info(f'\t[>] Begin data cleaning')
        df = df.copy()

        # Change the data
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.date

        # Remove rows with missing values
        df = df.dropna()

        # Remove the cities that are not city codes.
        mask = (
                df['origin_city'].str.len().le(3) &
                df['destination_city'].str.len().le(3)
        )
        df = df.loc[mask]

        return df


