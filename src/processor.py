from typing import Tuple
import pandas as pd
import numpy as np

class DataProcessor:
    """
    Comprehensive data processor for cargo market analysis.

    This class handles the complete data processing pipeline from raw WACD market data
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
        # Default to 2% threshold for trend stability if not specified in settings
        self.trend_threshold = settings.get('trend_threshold', 0.02)  # fallback to 2% if not set

        pass

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete data processing pipeline.

        This method orchestrates all processing steps in the correct sequence:
        - Data cleaning and validation
        - Product and route aggregation
        - Feature engineering for model training

        Args:
            df (pd.DataFrame): Raw WACD market data with columns for product, origin/destination cities,
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
        df_product, df_route = self._aggregate_market_data(df)
        df_feature = self._add_revenue_share_features(df_route)
        df_feature = self._add_seasonality_features(df_feature)
        df_feature = self._add_lagged_features(df_feature)
        df_feature = self._add_feature_ratios(df_feature)
        df_feature = self._classify_product_trends(df_feature)

        return df_feature

    def _clean_raw_data(self, df) -> pd.DataFrame:
        """
        Clean and validate raw market data for analysis.

        This method performs essential data quality checks and transformations:
        - Converts dates to monthly periods for consistent aggregation
        - Removes records with missing critical information
        - Filters out invalid city codes (non-3-letter codes)

        Args:
            df (pd.DataFrame): Raw WACD data

        Returns:
            pd.DataFrame: Cleaned dataset with valid monthly records only
        """
        self.logger.info(f'\t[>] Cleaning raw data - removing invalid records')
        df = df.copy()

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

    def _aggregate_market_data (self, df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate market data by product and route to calculate market shares.

        This method creates two levels of aggregation:
        1. Product-level: Total volumes by product and month
        2. Route-level: Volumes by product, origin-destination pair, and month

        The route-level data includes market share calculations showing what percentage
        of each route's total volume each product represents.

        Args:
            df (pd.DataFrame): Cleaned market data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - df_product: Product-level monthly aggregates
                - df_route: Route-level data with market share calculations
        """
        self.logger.info(f'\t[>] Aggregating market data by product and route')

        # Define the key benchmark metrics to aggregate
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

        # Calculate each product's weight share of the total route volume
        # This is the target variable for prediction
        df_route['weight_share'] = df_route['benchmark_actual_weight'] / df_route['total_weight']
        df_route['weight_share'] = df_route['weight_share'] .fillna(0)

        self.logger.info(f'\t    Created {len(df_product)} product records and {len(df_route)} route records')
        return df_product, df_route

    def _add_revenue_share_features(self, df_route: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate revenue-based market share features.

        While weight share shows volume distribution, revenue share reveals
        value distribution across products on each route. This provides insight
        into premium vs. commodity product positioning.

        Args:
            df_route (pd.DataFrame): Route-level aggregated data

        Returns:
            pd.DataFrame: Enhanced dataset with revenue share features
        """
        self.logger.info('\t[>] Computing revenue share features')

        df = df_route.copy()

        # Get the total actual and revenue weight on OD per month
        df['total_revenue'] = df.groupby(
            ['origin_city', 'destination_city', 'date']
        )['benchmark_revenue'].transform('sum')

        # Calculate each product's share of total route revenue
        df['share_revenue'] = df['benchmark_revenue'] / df['total_revenue']
        df['share_revenue'] = df['share_revenue'].fillna(0)

        return df

    def _add_seasonality_features(self, df_route: pd.DataFrame) -> pd.DataFrame:
        """
        Build time-based seasonality and trend features.

        This method adds multiple temporal features to capture seasonal patterns:
        - Month and quarter indicators for seasonal effects
        - Time index for linear trends
        - Rolling averages to smooth short-term volatility

        Args:
            df_route (pd.DataFrame): Route data with basic features

        Returns:
            pd.DataFrame: Dataset enhanced with seasonality features
        """
        self.logger.info('\t[>] Adding seasonality and rolling average features')

        df = df_route.copy()

        # Convert period back to timestamp for date operations
        df['date'] = df['date'].dt.to_timestamp()

        # Extract seasonal indicators
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # Create numeric time index for trend analysis
        # This converts dates to a simple sequence: Jan 2024=1, Feb 2024=2, etc.
        years = df['date'].dt.year
        df['t'] = (years - years.min()) * 12 + df['month']

        # Calculate 3-month rolling averages to capture recent trends
        # These smooth out monthly volatility and highlight underlying patterns
        group_cols = ['product', 'origin_city', 'destination_city']
        df = df.sort_values(group_cols + ['date'])

        # Rolling averages with minimum 1 period to handle early months
        for col_name, source_col in [
            ('ma3_share_wt', 'weight_share'),
            ('ma3_share_rev', 'share_revenue'),
            ('ma3_revenue', 'benchmark_revenue')
        ]:
            df[col_name] = df.groupby(group_cols)[source_col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        return df

    def _add_lagged_features(self, df_route: pd.DataFrame) -> pd.DataFrame:
        """
        Add previous month's performance indicators as predictive features.

        Lagged features capture momentum and recent performance trends,
        which are often strong predictors of next month's market share.
        These features help the model understand short-term dynamics.

        Args:
            df_route (pd.DataFrame): Dataset with seasonality features

        Returns:
            pd.DataFrame: Dataset with lagged performance indicators
        """
        self.logger.info('\t[>] Adding previous month performance indicators')

        df_lagged = df_route.copy()

        # Define grouping for time series operations
        grouping_columns = ['product', 'origin_city', 'destination_city']

        # Create lagged features (previous month's values)
        lag_features = {
            'lag1_actual_wt': 'benchmark_actual_weight',
            'lag1_chargeable_wt': 'benchmark_chargeable_weight',
            'lag1_revenue': 'benchmark_revenue',
            'lag1_share': 'weight_share'
        }

        for lag_col, source_col in lag_features.items():
            df_lagged[lag_col] = df_lagged.groupby(grouping_columns)[source_col].shift(1)

        # Fill NaN values (first month for each group) with 0
        df_lagged = df_lagged.fillna(0)

        return df_lagged

    def _add_feature_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ratio and interaction features for cargo profitability analysis.:
        - weighted_revenue: Market share weighted by revenue (popularity × profitability)
        - revenue_per_kg: Yield per actual weight (premium vs commodity indicator)
        - revenue_per_chargeable_kg: Yield per billed weight (pricing efficiency)
        - chargeability_ratio: Volumetric efficiency (dense vs voluminous cargo)

        Args:
            df (pd.DataFrame): Dataset with weight_share, revenue, and weight columns

        Returns:
            pd.DataFrame: Enhanced dataset with ratio features for model training
        """
        self.logger.info('\t[>] Building features: ratio and interaction metrics')

        df = df.copy()

        # Revenue weighted by product share in the route
        # A product might have high share but low value (or vice versa)
        # This reflects both popularity and profitability.
        df['weighted_revenue'] = df['weight_share'] * df['benchmark_revenue']

        # Revenue earned per kg of actual flown weight
        # Create masks for non-zero weights to avoid division by zero
        actual_weight_mask = df['benchmark_actual_weight'] > 0
        chargeable_weight_mask = df['benchmark_chargeable_weight'] > 0

        # Initialize yield columns with zeros
        df['revenue_per_kg'] = 0.0
        df['revenue_per_chargeable_kg'] = 0.0
        df['chargeability_ratio'] = 0.0

        # Calculate yields only where weights are positive (vectorized approach)
        df.loc[actual_weight_mask, 'revenue_per_kg'] = (
                df.loc[actual_weight_mask, 'benchmark_revenue'] /
                df.loc[actual_weight_mask, 'benchmark_actual_weight']
        )

        # Revenue earned per kg of chargeable (billed) weight
        # what the customer paid, not just what physically shipped (pricing-sensitive markets)
        df.loc[chargeable_weight_mask, 'revenue_per_chargeable_kg'] = (
                df.loc[chargeable_weight_mask, 'benchmark_revenue'] /
                df.loc[chargeable_weight_mask, 'benchmark_chargeable_weight']
        )

        # how “volumetric” a product is. A higher ratio means it's charged more than it weighs.
        # For space-consuming items that are not dense
        df.loc[actual_weight_mask, 'chargeability_ratio'] = (
                df.loc[actual_weight_mask, 'benchmark_chargeable_weight'] /
                df.loc[actual_weight_mask, 'benchmark_actual_weight']
        )

        return df

    def _classify_product_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify product trends for Spotfire dashabord business insight and visualization.

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

        if share_difference <= self.trend_threshold:
            return 'stable'
        elif current_share > historical_average:
            return 'growth'
        else:  # current_share < historical_average
            return 'decline'




