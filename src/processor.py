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
        self.trend_threshold = settings.get('trend_threshold', 0.02)  # fallback to 2% if not set

        pass

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all functions for data processing
        """
        self.logger.info('[>] Begin data processing')
        df = self._clean(df)
        df_product, df_route = self._prepare_data_product_demand(df)
        df_feature = self._build_feature_revenue(df_route)
        df_feature = self._build_feature_seasonality(df_feature)
        df_feature = self._build_feature_last_month(df_feature)
        df_feature = self._build_feature_ratios(df_feature)
        df_feature = self._add_product_trends(df_feature)

        return df_feature

    def _clean(self, df) -> pd.DataFrame:
        """
        Clean raw data: drop NAs, correct dtypes, etc.

        :return: Dataframe with only valid, clean, monthly-level records remain for modeling
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

    def _prepare_data_product_demand (self, df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        # Aggregate benchmark metrics by product x origin x destination x month
        df_route = (
            df
            .groupby(['product', 'origin_city', 'destination_city', 'date'], as_index=False)[bench_cols]
            .sum()
        )

        # Aggregate the total weight per O&D month
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
        Computes how much revenue each product earned as a share of total route revenue that month

        :return: Dataframe with share_revenue added
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
        Building seasonality features of Month, Quater, and rolling averages

        :return: DataFrame with added columns: month, quarter, ma3_share_wt, ma3_share_rev, ma3_revenue
        """
        self.logger.info('\t[>] Building features: seasonality')

        df = df_route.copy()

        df['date'] = df['date'].dt.to_timestamp()

        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # a simple numeric index for time (e.g. Jan 2024 → 1, Feb 2024 → 2, … Jan 2025 → 13)
        years = df['date'].dt.year
        df['t'] = (years - years.min()) * 12 + df['month']

        # 3-month rolling averages to capture recent trends and smooth volatility
        group_cols = ['product', 'origin_city', 'destination_city']
        df = df.sort_values(group_cols + ['date'])
        df['ma3_share_wt'] = df.groupby(group_cols)['weight_share'].transform(
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
            - Give model access to the immediate past — good for forecasting and detecting short-term momentum

        """
        self.logger.info('\t[>] Building features: Apply the data of previous month')

        df = df_route.copy()

        # for each product of O&D, see the difference of the current value from the past 1 month
        group_cols = ['product', 'origin_city', 'destination_city']
        df['lag1_actual_wt'] = df.groupby(group_cols)['benchmark_actual_weight'].shift(1)
        df['lag1_chargeable_wt'] = df.groupby(group_cols)['benchmark_chargeable_weight'].shift(1)
        df['lag1_revenue'] = df.groupby(group_cols)['benchmark_revenue'].shift(1)
        df['lag1_share'] = df.groupby(group_cols)['weight_share'].shift(1)
        df = df.fillna(0)

        return df

    def _build_feature_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio and interaction features:
            - share_x_revenue: interaction of weight share and revenue
            - revenue_per_kg: yield per actual weight (kg)
            - revenue_per_chargeable_kg: yield per billed weight (kg)
            - chargeability_ratio: volumetric efficiency of the cargo

        """
        self.logger.info('\t[>] Building features: ratio and interaction metrics')

        df = df.copy()

        # Revenue weighted by product share in the route
        # A product might have high share but low value (or vice versa)
        # This reflects both popularity and profitability.
        df['weighted_revenue'] = df['weight_share'] * df['benchmark_revenue']

        # Revenue earned per kg of actual flown weight
        # Replace 0 with NaN to avoid divide-by-zero, then fill resulting NaNs with 0
        df['revenue_per_kg'] = df['benchmark_revenue'] / df['benchmark_actual_weight'].replace(0, np.nan)
        df['revenue_per_kg'] = df['revenue_per_kg'].fillna(0)

        # Revenue earned per kg of chargeable (billed) weight
        # what the customer paid, not just what physically shipped (pricing-sensitive markets)
        df['revenue_per_chargeable_kg'] = df['benchmark_revenue'] / df['benchmark_chargeable_weight'].replace(0, np.nan)
        df['revenue_per_chargeable_kg'] = df['revenue_per_chargeable_kg'].fillna(0)

        # how “volumetric” a product is. A higher ratio means it's charged more than it weighs.
        # For space-consuming items that are not dense
        df['chargeability_ratio'] = df['benchmark_chargeable_weight'] / df['benchmark_actual_weight'].replace(0, np.nan)
        df['chargeability_ratio'] = df['chargeability_ratio'].fillna(0)

        return df

    def _add_product_trends(self, df):
        """
        This function is used to get a product trend and only used for a visualization with historic data.

        This identifies whether a product is showing 'growth', 'not_present', 'decline', 'stable', 'new',
        or 'disappeared' behavior, by comparing the current month's share to the average of past 3 active (non-zero)
        months.

        :param df: DataFrame containing weight_share per product × O&D × date
        :return: DataFrame with added columns: weight_share_ma3, trend, active_count_ma3
        """
        self.logger.info('\t[>] Adding product trends')

        df = df.sort_values(['origin_city', 'destination_city', 'product', 'date'])

        # Exclude past months with 0 weight share when computing the rolling average (=.shift(1)),
        # and take the rolling average of 3 periods (previous 3 active months)
        df['weight_share_ma3'] = (
            df.groupby(['origin_city', 'destination_city', 'product'])['weight_share']
            .transform(lambda x: x.shift(1).where(x.shift(1) > 0).rolling(3, min_periods=1).mean())
        )

        df['product_trend'] = df.apply(
            lambda row: self._classify_trend(row['weight_share'], row['weight_share_ma3']), axis=1
        )

        # how often a product was active in the past
        df['active_count_ma3'] = (
            df.groupby(['origin_city', 'destination_city', 'product'])['weight_share']
            .transform(lambda x: x.shift(1).gt(0).rolling(3).sum())
        )

        return df

    def _classify_trend(self, current, past_avg):
        """
        Classify trend comparing current weight_share to 3-month average.
        Threshold defines tolerance for 'stable'.

        :param current: Current month's weight share
        :param past_avg: 3-month rolling average of weight share
        :return: trend label
        """
        if pd.isna(past_avg):
            return 'new' if current > 0 else 'not_present'

        elif current == 0 and past_avg > 0:
            return 'disappeared'

        elif abs(current - past_avg) <= self.trend_threshold:
            return 'stable'

        elif current > past_avg:
            return 'growth'

        elif current < past_avg:
            return 'decline'





