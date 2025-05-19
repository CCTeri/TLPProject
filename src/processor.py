from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor(object):
    """
    Process the data

    :arg
        settings (dict): Project settings
        logger (logging.Logger): Logger object
    """

    def __init__(self, settings, logger, df: pd.DataFrame):
        self.settings = settings
        self.logger = logger

        self.df = df.copy()

        pass

    def process_data(self, df):
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

        # Change the data
        df['date'] = pd.to_datetime(df['date'])

        # Remove the cities that are not city codes.
        mask = (
                df['origin_city'].str.len().le(3) &
                df['destination_city'].str.len().le(3)
        )
        df = df.loc[mask]

        return df




    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create any new features (e.g. lagged share, yearly growth rate).
        """
        df = df.sort_values(['product_id', 'date'])
        df['share_lag_1'] = df.groupby('product_id')['market_share'].shift(1)
        df['growth_rate'] = (df['market_share'] - df['share_lag_1']) / df['share_lag_1']
        df = df.dropna(subset=['share_lag_1'])
        return df

    def _split(self, df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42
             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split into train/test sets.
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def _scale(self, X_train, X_test):
        """
        Scale numeric features.
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include='number'))
        X_test_scaled  = scaler.transform(X_test.select_dtypes(include='number'))
        # If you have categorical cols, you'll need to handle them separately
        return X_train_scaled, X_test_scaled
