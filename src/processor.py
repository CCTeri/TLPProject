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
        df_product, df_route = self._prepare_data_product_demand(df)
        df_feature = self._build_feature_revenue(df_route)
        # df_feature = self._build_feature_seasonality(df_feature)
        # df_feature = self._build_feature_last_month(df_feature)

        return df_feature

    def _clean(self, df) -> pd.DataFrame:
        """
        Clean raw data: drop NAs, correct dtypes, etc.
        """
        self.logger.info(f'\t[>] Begin data cleaning')
        df = df.copy()

        # Change the data
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M')

        # Remove rows with missing values
        df = df.dropna(subset=[
            'product',
            'origin_city',
            'destination_city',
            'date'
        ])

        # Remove the cities that are not city codes.
        mask = (
                df['origin_city'].str.len().le(3) &
                df['destination_city'].str.len().le(3)
        )
        df = df.loc[mask]

        return df

    def _prepare_data_product_demand (self, df:pd.DataFrame)-> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the product and demand data

        :param df: the WACD market data
        :return:
            - df_product: product x month total
            - df_route: product x O&D x month total
        """
        self.logger.info(f'\t[>] Prepare the product and route demand data')

        bench_cols = [
            'benchmark_actual_weight',
            'benchmark_chargeable_weight',
            'benchmark_revenue'
        ]

        # Aggregate: Product Level
        df_product = (
            df
            .groupby(['product', 'date'], as_index=False)[bench_cols]
            .sum()
        )

        # Check which market is rising a certain product
        df_route = (
            df
            .groupby(['product', 'origin_city', 'destination_city', 'date'], as_index=False)[bench_cols]
            .sum()
        )

        df_route['total_weight'] = (
            df_route
            .groupby(['origin_city', 'destination_city', 'date'])['benchmark_actual_weight']
            .transform('sum')
        )

        # Weight Share = what is the % of weight share does the product have for the OD pair
        df_route['weight_share'] = df_route['benchmark_actual_weight'] / df_route['total_weight']
        df_route['weight_share'] = df_route['weight_share'] .fillna(0)

        return df_product, df_route

    def _build_feature_revenue(self, df_route: pd.DataFrame) -> pd.DataFrame:
        """
        Building features
        """
        self.logger.info('\t[>] Building features: actual weight')

        df = df_route.copy()

        # Get the total actual and revenue weight on OD per month
        df['total_revenue'] = df.groupby(
            ['origin_city', 'destination_city', 'date']
        )['benchmark_revenue'].transform('sum')

        # Get the share of the product
        df['share_revenue'] = df['benchmark_revenue'] / df['total_revenue']
        df['share_revenue'] = df['share_revenue'].fillna(0)

        return df

    def _build_feature_seasonality(self, df_route: pd.DataFrame) -> pd.DataFrame:
        """
        Building features
        """
        self.logger.info('\t[>] Building features: seasonality')

        df = df_route.copy()

        df['month']   = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # a simple numeric index for time (e.g. Jan 2024 → 1, Feb 2024 → 2, …)
        df['t'] = (
            (df['date'].dt.year - df['date'].dt.year.min()) * 12
            + df['date'].dt.month
        )

        # 3-month rolling averages (optional)
        group_cols = ['product', 'origin_city', 'destination_city']
        df['ma3_share_wt'] = df.groupby(group_cols)['share_actual'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df['ma3_share_rev'] = df.groupby(group_cols)['share_revenue'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df['ma3_revenue'] = df.groupby(group_cols)['benchmark_revenue'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())

        return df

    def _build_feature_last_month(self, df_route: pd.DataFrame) -> pd.DataFrame:
        """
        for each row representing a given (product, origin_city, destination_city, date),
        we pull in four numbers from the previous month on that same series.
            - Without lag features, the model only sees static attributes (route, product, month)
              and can’t learn patterns like “this product usually jumps after a big prior-month volume.”

        """
        self.logger.info('\t[>] Building features: Apply the data of previous month')

        df = df_route.copy()

        group_cols = ['product', 'origin_city', 'destination_city']
        df['lag1_actual_wt'] = df.groupby(group_cols)['benchmark_actual_weight'].shift(1)
        df['lag1_chargeable_wt'] = df.groupby(group_cols)['benchmark_chargeable_weight'].shift(1)
        df['lag1_revenue'] = df.groupby(group_cols)['benchmark_revenue'].shift(1)
        df['lag1_share'] = df.groupby(group_cols)['share_actual'].shift(1)
        df = df.fillna(0)

        return df



