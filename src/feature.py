import pandas as pd
import logging


class FeatureEngineer:
    """
    Handles all feature engineering operations for the TLP Project.
    """

    def __init__(self, settings: dict, logger: logging.Logger):
        """
        Initialize the Feature Engineer with settings and logger.

        Args:
            settings: Configuration dictionary
            logger: Logger instance for tracking operations
        """
        self.settings = settings
        self.logger = logger
        self.trend_threshold = settings.get('trend_threshold', 0.02)

    def build_features(self, df_route: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to create all features.

        Args:
            df_route: DataFrame with basic route data

        Returns:
            DataFrame with all features added
        """
        self.logger.info('[>] Starting feature creation pipeline')

        # Basic features
        df_feature = self._add_revenue_share_features(df_route)
        df_feature = self._add_seasonality_features(df_feature)
        df_feature = self._add_lagged_features(df_feature)
        df_feature = self._add_feature_ratios(df_feature)

        # Calculated features
        df_feature = self._build_feature_market_competition(df_feature)
        df_feature = self._build_feature_route_characteristics(df_feature)
        df_feature = self._build_feature_cross_route_patterns(df_feature)

        return df_feature

    def get_modeling_features(self):
        """
        Returns the list of features for demand forecasting.
        Call this method in Modeler to get the feature list.

        Returns:
            list: Feature names for modeling
        """
        features = [
            # Previous performance
            'lag1_share',  # Last month's market share
            'ma3_share_wt',  # 3-month moving average

            # Seasonality
            'month',  # Monthly patterns (1-12)
            'quarter',  # Quarterly patterns (1-4)
            't',  # Time trend

            # Route characteristics
            'route_total_volume',  # Market size
            'route_diversity',  # Number of competing products
            'route_growth_trend',  # Route momentum

            # Competition
            'market_concentration',  # How concentrated is the market?
            'product_rank',  # Current ranking
            'is_market_leader',  # Leader flag
            'top3_concentration',  # Top 3 combined share

            # Cross-route patterns
            'origin_product_strength',  # Product strength from origin
            'destination_product_strength',  # Product strength to destination
            'product_avg_share',  # Overall product performance

        ]

        self.logger.info(f"Selected {len(features)} features for demand forecasting")
        return features

    def _add_revenue_share_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate revenue-based market share features.

        While weight share shows volume distribution, revenue share reveals
        value distribution across products on each route. This provides insight
        into premium vs. commodity product positioning.

        Args:
            df (pd.DataFrame): Route-level aggregated data

        Returns:
            pd.DataFrame: Enhanced dataset with revenue share features
        """
        self.logger.info('\t[>] Computing revenue share features')

        # Get the total actual and revenue weight on OD per month
        df['total_revenue'] = df.groupby(
            ['origin_city', 'destination_city', 'date']
        )['benchmark_revenue'].transform('sum')

        # Calculate each product's share of total route revenue
        df['share_revenue'] = df['benchmark_revenue'] / df['total_revenue']
        df['share_revenue'] = df['share_revenue'].fillna(0)

        return df

    def _add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build time-based seasonality and trend features.

        This method adds multiple temporal features to capture seasonal patterns:
        - Month and quarter indicators for seasonal effects
        - Time index for linear trends
        - Rolling averages to smooth short-term volatility

        Args:
            df (pd.DataFrame): Route data with basic features

        Returns:
            pd.DataFrame: Dataset enhanced with seasonality features
        """
        self.logger.info('\t[>] Adding seasonality and rolling average features')

        # Convert period back to timestamp for date operations
        df['date'] = df['date'].dt.to_timestamp()
        date_dt = df['date'].dt

        df['month'] = date_dt.month
        df['quarter'] = date_dt.quarter

        # Create numeric time index for trend analysis
        # This converts dates to a simple sequence: Jan 2024=1, Feb 2024=2, etc.
        df['t'] = (date_dt.year - date_dt.year.min()) * 12 + df['month']

        # Calculate 3-month rolling averages to capture recent trends
        # These smooth out monthly volatility and highlight underlying patterns
        group_cols = ['product', 'origin_city', 'destination_city']
        df = df.sort_values(group_cols + ['date'])
        grouped = df.groupby(group_cols)

        # Rolling averages with minimum 1 period to handle early months
        def rolling_mean_3m(series):
            return series.rolling(window=3, min_periods=1).mean()

        rolling_features = {
            'ma3_share_wt': grouped['weight_share'].transform(rolling_mean_3m),
            'ma3_share_rev': grouped['share_revenue'].transform(rolling_mean_3m),
            'ma3_revenue': grouped['benchmark_revenue'].transform(rolling_mean_3m)
        }

        df = df.assign(**rolling_features)

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add previous month's performance indicators as predictive features.

        Lagged features capture momentum and recent performance trends,
        which are often strong predictors of next month's market share.
        These features help the model understand short-term dynamics.

        Args:
            df (pd.DataFrame): Dataset with seasonality features

        Returns:
            pd.DataFrame: Dataset with lagged performance indicators
        """
        self.logger.info('\t[>] Adding previous month performance indicators')

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
            df[lag_col] = df.groupby(grouping_columns)[source_col].shift(1)

        # Fill NaN values (first month for each group) with 0
        df = df.fillna(0)

        return df

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
        self.logger.info('\t[>] Adding ratio and interaction metrics')

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

    def _build_feature_market_competition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market competition and concentration features.

        These features help the model understand competitive dynamics:
        - Understand market context, not just product history.
        - Learn that products in competitive markets behave differently than in monopolistic ones.
        - Detect leader advantage (e.g., first-ranked products tend to retain share).

        Returns features:
        - market_concentration: Herfindahl index (0=very competitive, 1=monopoly)
        - product_rank: This product's rank by weight share on this route
        - top3_concentration: Combined share of top 3 products
        - is_market_leader: Boolean if this product has the highest share
        """
        self.logger.info('\t[>] Adding market competition metrics')

        # Calculate market concentration using Herfindahl-Hirschman Index (If many products have equal share, the HHI is low (e.g., 0.1–0.3).)
        # Sum of squared market shares - higher = more concentrated market, less competition
        df['market_concentration'] = (
            df.groupby(['origin_city', 'destination_city', 'date'])['weight_share']
            .transform(lambda x: (x ** 2).sum())
        )

        # Rank products by weight share within each route-month. "dense" = ties will get the same rank
        df['product_rank'] = (
            df.groupby(['origin_city', 'destination_city', 'date'])['weight_share']
            .rank(method='dense', ascending=False)
        )

        # Calculate combined share of top 3 products (market concentration indicator) - High value = small number of players dominate the market.
        top3_shares = (
            df.groupby(['origin_city', 'destination_city', 'date'])['weight_share']
            .transform(lambda x: x.nlargest(3).sum())
        )
        df['top3_concentration'] = top3_shares

        # Flag if this product is the market leader (Binary feature for leadership status — models can learn different behavior for leaders.)
        df['is_market_leader'] = (df['product_rank'] == 1).astype(int)

        return df

    def _build_feature_route_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create route-level characteristics that affect product demand.

        Different routes have different characteristics that favor certain products:
        - Route maturity (how long has it been active?)
        - Route size (total volume)
        - Route diversity (how many products compete?)
        - Route growth trend

        Returns features:
        - route_total_volume: Total weight across all products on this route
        - route_diversity: Number of active products on this route
        - route_maturity: Number of months this route has been active
        - route_growth_trend: 6-month trend in total route volume
        """
        self.logger.info('\t[>] Adding route characteristics')

        df_route = df.sort_values(['origin_city', 'destination_city', 'date'])

        # Define route grouping columns once
        route_cols = ['origin_city', 'destination_city']
        route_date_cols = route_cols + ['date']

        # Total flown weight on the route in that month; larger routes might attract more stable or competitive product dynamics
        route_totals = (
            df_route.groupby(route_date_cols, sort=False)['benchmark_actual_weight']
            .sum()
            .rename('route_total_volume')
        )
        df_route = df_route.merge(route_totals, left_on=route_date_cols, right_index=True, how='left')

        # Route diversity - number of products with >1% market share: For each route-month, counts how many products had more than 1% share.
        df_route['route_diversity'] = (
            df_route.groupby(route_date_cols, sort=False)['weight_share']
            .transform(lambda x: (x > 0.01).sum())
        )

        # Route maturity - How long the route has been active?
        route_maturity = (
            df_route.groupby(route_cols, sort=False)['date']
            .agg(lambda x: (x.max() - x.min()).days / 30)
            .rename('route_maturity')
        )
        df_route = df_route.merge(route_maturity, left_on=route_cols, right_index=True, how='left')

        # Route growth trend - For each route, calculates the 6-month percentage change in total volume
        df_route['route_growth_trend'] = (
            df_route.groupby(['origin_city', 'destination_city'])['route_total_volume']
            .transform(lambda x: x.pct_change(periods=6))
        )

        # Fill NaN values with 0
        feature_cols = ['route_total_volume', 'route_diversity', 'route_maturity', 'route_growth_trend']
        df_route[feature_cols] = df_route[feature_cols].fillna(0)

        return df_route

    def _build_feature_cross_route_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that compare product performance across similar routes, cities, or market sizes.
        These help the model detect global product behavior, not just route-specific trends.

        These features help identify products that are strong in similar markets:
        - How does this product perform on similar routes?
        - Are there regional patterns?
        - Cross-route momentum indicators

        Returns features:
        - product_avg_share: This product's average share across all routes
        - similar_routes_performance: Average performance on routes with similar characteristics
        """
        self.logger.info('\t[>] Adding cross-route patterns')

        # Product's overall average market share across all routes (Tells the model how strong the product is overall)
        df['product_avg_share'] = (
            df.groupby(['product', 'date'])['weight_share']
            .transform('mean')
        )

        # Performance on similar-sized routes (routes with similar total volume)
        # For this product, what is the average weight_share on routes of similar size (quartile), on this date?
        # First, create route size categories
        df['route_size_quartile'] = (
            df.groupby('date')['route_total_volume']
            .transform(lambda x: pd.qcut(x, q=4, labels=[1, 2, 3, 4], duplicates='drop'))
        )

        # Then, average performance on routes of similar size (The average share of this product across routes in the same volume quartile.)
        df['similar_routes_performance'] = (
            df.groupby(['product', 'route_size_quartile', 'date'], observed=True)['weight_share']
            .transform('mean')
        )

        # Compare the product of the market with benchmark
        df['share_vs_peers'] = df['weight_share'] - df['similar_routes_performance']

        # Fill any remaining NaN values
        feature_cols = ['product_avg_share', 'similar_routes_performance', 'share_vs_peers']
        df[feature_cols] = df[feature_cols].fillna(0)

        return df


