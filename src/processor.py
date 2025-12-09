import pandas as pd
import numpy as np

class DataProcessor:
    """
    Comprehensive data processor for cargo market analysis.

    This class handles the complete data processing pipeline from raw market data
    to model-ready features for predicting product market share across origin-destination routes.

    The processing pipeline includes:
    1. Data cleaning and validation
    2. Aggregation by product and route
    3. Market share calculations
    4. Feature engineering (seasonality, trends, lags)
    5. Product trend classification

    Attributes:
        settings (dict): Configuration parameters for processing
        logger (logging.Logger): Logger for tracking processing steps
        trend_threshold (float): Threshold for classifying product trends as stable vs growing/declining
    """

    def __init__(self, settings: dict, logger):
        """
        Initialize the DataProcessor with configuration settings.

        Args:
            settings (dict): Project configuration containing processing parameters
            logger (logging.Logger): Logger instance for tracking operations
        """
        self.settings = settings
        self.logger = logger

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete data processing pipeline.

        This method orchestrates all processing steps in the correct sequence:
        - Data cleaning and validation
        - Product and route aggregation
        - Feature engineering for model training

        Args:
            df (pd.DataFrame): Raw market data with columns for product, origin/destination cities,
                             dates, and benchmark metrics (weight, revenue)

        Returns:
            pd.DataFrame: Processed dataset ready for machine learning with features including:
                        - Market share percentages
                        - Seasonality indicators
                        - Lagged values
                        - Product trend classifications
        """
        self.logger.info('[>] Begin data processing')
        df = self._clean_raw_data(df)
        df_route = self._aggregate_market_data(df)
        df_route = self._classify_product_trends(df_route)

        return df_route

    def _clean_raw_data(self, df) -> pd.DataFrame:
        """
        Clean and validate raw market data for analysis.

        This method performs essential data quality checks and transformations:
        - Converts dates to monthly periods for consistent aggregation
        - Removes records with missing critical information
        - Filters out invalid city codes (non-3-letter codes)

        Args:
            df (pd.DataFrame): Raw market data

        Returns:
            pd.DataFrame: Cleaned dataset with valid monthly records only
        """
        self.logger.info(f'\t[>] Cleaning raw data - removing invalid records')

        # Convert dates to monthly periods for consistent time-series analysis
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M')

        # Remove rows with missing values
        df = df.dropna(subset=[
            'product',
            'origin_city',
            'destination_city',
            'date'
        ])

        # Filter for valid 3-letter city codes only
        mask = (
                df['origin_city'].str.len().le(3) &
                df['destination_city'].str.len().le(3)
        )
        df = df.loc[mask]

        return df

    def _aggregate_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate market data by route to calculate market shares.

        Creates route-level aggregation with market share calculations showing what
        percentage of each route's total volume each product represents.

        Args:
            df: Cleaned market data

        Returns:
            Route-level data with market share calculations
        """
        self.logger.info(f'\t[>] Aggregating market data by product and route')

        # Define the key benchmark metrics to aggregate
        bench_cols = [
            'benchmark_actual_weight',
            'benchmark_chargeable_weight',
            'benchmark_revenue'
        ]

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

        df_route = df_route[df_route['total_weight'] > self.settings['market_size']]

        # Calculate each product's weight share of the total route volume
        # This is the target variable for prediction
        df_route['weight_share'] = df_route['benchmark_actual_weight'] / df_route['total_weight']
        df_route['weight_share'] = df_route['weight_share'].fillna(0)

        return df_route

    def _classify_product_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify product trends for Spotfire dashboard business insight and visualization.

        This method identifies whether each product is experiencing growth, decline,
        stability, or market entry/exit patterns. The classification compares current
        market share to historical average performance.

        Trend Categories:
        - 'new': Product has current share but no historical presence
        - 'not_present': Product has no current or historical presence
        - 'disappeared': Product had historical presence but zero current share
        - 'stable': Current share within threshold of historical average
        - 'growth': Current share significantly above historical average
        - 'decline': Current share significantly below historical average

        Args:
            df (pd.DataFrame): Dataset with all previous features

        Returns:
            pd.DataFrame: Final dataset with trend classifications
        """
        self.logger.info('\t[>] Classifying product market trends')

        df = df.sort_values(['origin_city', 'destination_city', 'product', 'date'])

        # Exclude past months with 0 weight share when computing the rolling average (=.shift(1)),
        # and take the rolling average of 3 periods (previous 3 active months)
        df['weight_share_ma3'] = (
            df.groupby(['origin_city', 'destination_city', 'product'])['weight_share']
            .transform(lambda x: x.shift(1).where(x.shift(1) > 0).rolling(3, min_periods=1).mean())
        )

        # Apply trend classification logic
        df['product_trend'] = df.apply(
            lambda row: self._determine_trend_category(row['weight_share'], row['weight_share_ma3']), axis=1
        )

        # Count how often a product was active in the past
        df['active_count_ma3'] = (
            df.groupby(['origin_city', 'destination_city', 'product'])['weight_share']
            .transform(lambda x: x.shift(1).gt(0).rolling(3).sum())
        )

        return df

    def _determine_trend_category(self, current_share: float, historical_average: float) -> str:
        """
        Determine the trend category for a product based on current vs historical performance.

        This helper method implements the business logic for trend classification
        using the configured threshold for stability determination.

        Args:
            current_share (float): Current month's market share
            historical_average (float): 3-month rolling average of historical shares

        Returns:
            str: Trend category ('new', 'not_present', 'disappeared', 'stable', 'growth', 'decline')
        """
        # Handle cases with no historical data
        if pd.isna(historical_average):
            return 'new' if current_share > 0 else 'not_present'

        # Product disappeared from market
        if current_share == 0 and historical_average > 0:
            return 'disappeared'

        # Compare current performance to historical baseline
        share_difference = abs(current_share - historical_average)

        if share_difference <= self.settings['trend_threshold']:
            return 'stable'
        elif current_share > historical_average:
            return 'growth'
        else:  # current_share < historical_average
            return 'decline'
